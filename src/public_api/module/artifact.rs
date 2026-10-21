//! Versioned postorder DAG. Sources and retained compiled definitions are interned separately.
use super::*;
use bincode::Options;
use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
const MAGIC: &[u8; 8] = b"GLRASM01";
const MAX_BYTES: u64 = 1 << 30;

#[derive(Serialize, Deserialize)]
struct Vocabulary {
    #[serde(deserialize_with = "artifact_limits::items")]
    entries: Vec<(u32, Vec<u8>)>,
    #[serde(deserialize_with = "artifact_limits::items")]
    exact_only: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct Exact {
    target: u32,
    #[serde(deserialize_with = "artifact_limits::items")]
    ranges: Vec<(u32, u32)>,
}

#[derive(Serialize, Deserialize)]
enum Source {
    Ebnf(String),
    Glrm(String),
    Json(String),
    Lark { text: String, llm: bool, tokens: BTreeMap<String, Vec<u32>> },
    Token(u32),
    Exact(Exact),
}

#[derive(Serialize, Deserialize)]
enum Value {
    Module(u32),
    Token(u32),
    Exact(Exact),
}

#[derive(Serialize, Deserialize)]
struct Instance {
    source: u32,
    mask_strategy: u8,
    boundary_mask_strategy: u8,
    #[serde(deserialize_with = "artifact_limits::items")]
    prepared: Vec<(u8, u32)>,
    #[serde(deserialize_with = "artifact_limits::items")]
    bindings: Vec<(String, Value)>,
}

#[derive(Serialize, Deserialize)]
struct Preparation {
    source: u32,
    engine: u8,
    boundary: u8,
    bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    #[serde(deserialize_with = "artifact_limits::nodes")]
    vocabs: Vec<Vocabulary>,
    #[serde(deserialize_with = "artifact_limits::nodes")]
    sources: Vec<Source>,
    #[serde(deserialize_with = "artifact_limits::nodes")]
    prepared: Vec<Preparation>,
    #[serde(deserialize_with = "artifact_limits::nodes")]
    instances: Vec<Instance>,
}

fn err(message: impl Into<String>) -> Error { Error::Serialization(message.into()) }

fn engine_to_u8(s: MaskStrategy) -> u8 {
    match s {
        MaskStrategy::Auto => 0,
        MaskStrategy::ParserDwa => 1,
        MaskStrategy::TokenTrie => 2,
        MaskStrategy::TokenClassTrie => 3,
    }
}

fn u8_to_engine(n: u8) -> Result<MaskStrategy> {
    match n {
        0 => Ok(MaskStrategy::Auto),
        1 => Ok(MaskStrategy::ParserDwa),
        2 => Ok(MaskStrategy::TokenTrie),
        3 => Ok(MaskStrategy::TokenClassTrie),
        _ => Err(err("invalid mask strategy tag")),
    }
}

fn boundary_to_u8(b: BoundaryMaskStrategy) -> u8 {
    match b {
        BoundaryMaskStrategy::FastBuild => 0,
        BoundaryMaskStrategy::FastRuntime => 1,
    }
}

fn u8_to_boundary(n: u8) -> Result<BoundaryMaskStrategy> {
    match n {
        0 => Ok(BoundaryMaskStrategy::FastBuild),
        1 => Ok(BoundaryMaskStrategy::FastRuntime),
        _ => Err(err("invalid boundary mask strategy tag")),
    }
}

fn vocab_id(vocab: &Vocab, vocabs: &mut Vec<Vocab>) -> u32 {
    if let Some(id) = vocabs.iter().position(|v| v.same_model_vocab(vocab)) { return id as u32; }
    let id = vocabs.len() as u32;
    vocabs.push(vocab.clone());
    id
}

fn exact(tokens: &ExactTokens, vocabs: &mut Vec<Vocab>) -> Exact {
    Exact { target: vocab_id(tokens.target_vocab(), vocabs), ranges: tokens.range_pairs().to_vec() }
}

fn source(grammar: &Grammar<'_>, vocabs: &mut Vec<Vocab>) -> Source {
    if let Some(t) = &grammar.exact_value { return Source::Exact(exact(t, vocabs)); }
    if let Some(id) = grammar.exact_token_id { return Source::Token(id); }
    match &grammar.source {
        GrammarSource::Ebnf(s) => Source::Ebnf(s.to_string()),
        GrammarSource::Glrm(s) => Source::Glrm(s.to_string()),
        GrammarSource::JsonSchema(s) => Source::Json(s.to_string()),
        GrammarSource::Lark(s) => Source::Lark {
            text: s.to_string(),
            llm: grammar.lark_options.profile == LarkProfile::Llm,
            tokens: grammar.lark_options.token_names.clone(),
        },
        GrammarSource::Constructed => Source::Glrm(grammar.component().expect("validated constructed source")
            .to_glrm_source().expect("serializable constructed source")),
    }
}

pub(super) fn save(root: &Module) -> Vec<u8> {
    let modules = root.index().expect("module graph validated on mutation");
    let mut source_ids = HashMap::new();
    let mut prepared_ids = HashMap::new();
    let mut instance_ids = HashMap::new();
    let mut sources = Vec::new();
    let mut prepared = Vec::new();
    let mut instances = Vec::new();
    let mut vocabs = Vec::new();

    for module in modules {
        let node = &module.node;
        let source_key = Arc::as_ptr(&node.source.parsed_component) as usize;
        let source_id = *source_ids.entry(source_key).or_insert_with(|| {
            let id = sources.len() as u32;
            sources.push(source(&node.source, &mut vocabs));
            id
        });
        let mut preps = Vec::new();
        for (&engine, p) in &node.prepared {
            let engine_u8 = engine_to_u8(engine);
            let prep_key = Arc::as_ptr(p) as *const () as usize;
            let prep_id = *prepared_ids.entry(prep_key).or_insert_with(|| {
                let id = prepared.len() as u32;
                prepared.push(Preparation {
                    source: source_id,
                    engine: engine_u8,
                    boundary: p.prepared().boundary_trigger as u8,
                    bytes: p.prepared().save(),
                });
                id
            });
            preps.push((engine_u8, prep_id));
        }
        let bindings = node.bindings.iter().map(|(name, value)| {
            let value = match value {
                BoundValue::Module(m) => Value::Module(instance_ids[&(Arc::as_ptr(&m.node) as usize)]),
                BoundValue::Token(t) => Value::Token(t.id()),
                BoundValue::ExactTokens(t) => Value::Exact(exact(t, &mut vocabs)),
            };
            (name.clone(), value)
        }).collect();
        instance_ids.insert(Arc::as_ptr(node) as usize, instances.len() as u32);
        instances.push(Instance {
            source: source_id,
            mask_strategy: engine_to_u8(node.mask_strategy),
            boundary_mask_strategy: boundary_to_u8(node.boundary_mask_strategy),
            prepared: preps,
            bindings,
        });
    }
    let manifest = Manifest {
        vocabs: vocabs.iter().map(|v| Vocabulary { entries: v.iter().map(|(id, b)| (id, b.to_vec())).collect(), exact_only: v.exact_only_token_ids().collect() }).collect(),
        sources,
        prepared,
        instances,
    };
    let body = bincode::serialize(&manifest).expect("module manifest serializable");
    let mut bytes = Vec::with_capacity(16 + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
    bytes.extend(body);
    bytes
}

fn decode_exact(value: Exact, vocabs: &[Vocab]) -> Result<ExactTokens> {
    let vocab = vocabs.get(value.target as usize).ok_or_else(|| err("invalid exact-event target reference"))?;
    let mut previous = None;
    for &(a, b) in &value.ranges {
        if a > b || previous.is_some_and(|p| u64::from(p) + 1 >= u64::from(a)) { return Err(err("unnormalized exact-event ranges")); }
        previous = Some(b);
    }
    vocab.token_ranges(value.ranges.into_iter().map(|(a, b)| a..=b)).map_err(|e| err(e.to_string()))
}

fn decode_source(value: Source, vocabs: &[Vocab]) -> Result<Grammar<'static>> {
    let source = match value {
        Source::Ebnf(s) => GrammarSource::Ebnf(SourceText::Owned(Arc::from(s))),
        Source::Glrm(s) => GrammarSource::Glrm(SourceText::Owned(Arc::from(s))),
        Source::Json(s) => GrammarSource::JsonSchema(SourceText::Owned(Arc::from(s))),
        Source::Lark { text, llm, tokens } => {
            let mut grammar = Grammar::new(GrammarSource::Lark(SourceText::Owned(Arc::from(text))));
            grammar.lark_options = Arc::new(LarkOptions { profile: if llm { LarkProfile::Llm } else { LarkProfile::Recognition }, token_names: tokens });
            return Ok(grammar);
        }
        Source::Token(id) => return Ok(Grammar::exact_token(Token::new(id))),
        Source::Exact(e) => return Ok(Grammar::from_exact_tokens(decode_exact(e, vocabs)?)),
    };
    Ok(Grammar::new(source))
}

pub(super) fn load(data: &[u8], supplied: Option<&Vocab>) -> Result<Module> {
    if data.len() < 16 || &data[..8] != MAGIC { return Err(err("invalid or unsupported Module artifact version")); }
    let size = u64::from_le_bytes(data[8..16].try_into().unwrap());
    if size > MAX_BYTES || size != data.len() as u64 - 16 { return Err(err("invalid or excessive Module artifact length")); }
    let manifest: Manifest = bincode::DefaultOptions::new().with_fixint_encoding().with_limit(size).reject_trailing_bytes()
        .deserialize(&data[16..]).map_err(|e| err(format!("invalid Module manifest: {e}")))?;
    if manifest.instances.is_empty() || manifest.instances.len() > graph::MAX_NODES || manifest.sources.is_empty()
        || manifest.sources.len() > graph::MAX_NODES || manifest.prepared.len() > graph::MAX_NODES || manifest.vocabs.len() > graph::MAX_NODES {
        return Err(err("invalid or excessive Module graph size"));
    }

    let mut used_vocabs = BTreeSet::new();
    for source in &manifest.sources { if let Source::Exact(e) = source { used_vocabs.insert(e.target); } }
    for node in &manifest.instances { for (_, value) in &node.bindings { if let Value::Exact(e) = value { used_vocabs.insert(e.target); } } }
    if used_vocabs.len() != manifest.vocabs.len() || used_vocabs.iter().any(|&id| id as usize >= manifest.vocabs.len()) {
        return Err(err("invalid or unreferenced exact-event vocabulary"));
    }

    let mut heights = Vec::new();
    let mut edges = Vec::new();
    let mut used_sources = BTreeSet::new();
    let mut used_prepared = BTreeSet::new();
    let mut ports = 0usize;

    for (i, node) in manifest.instances.iter().enumerate() {
        if node.source as usize >= manifest.sources.len() || node.mask_strategy > 3 || node.boundary_mask_strategy > 1 {
            return Err(err("invalid Module source/options"));
        }
        used_sources.insert(node.source);
        for &(engine_u8, p) in &node.prepared {
            let retained = manifest.prepared.get(p as usize).ok_or_else(|| err("invalid retained preparation reference"))?;
            if retained.source != node.source || retained.engine != engine_u8 {
                return Err(err("retained Module definition/options mismatch"));
            }
            used_prepared.insert(p);
        }
        ports = ports.checked_add(node.bindings.len()).ok_or_else(|| err("Module binding overflow"))?;
        if ports > 1_048_576 { return Err(err("too many Module bindings")); }
        let mut names = BTreeSet::new();
        let mut children = Vec::new();
        let mut height = 0;
        for (name, value) in &node.bindings {
            if name.is_empty() || !names.insert(name) { return Err(err("empty or duplicate Module binding")); }
            if let Value::Module(child) = value {
                let child = *child as usize;
                if child >= i { return Err(err("cyclic or forward Module reference")); }
                height = height.max(heights[child] + 1);
                children.push(child);
            }
        }
        if height > graph::MAX_DEPTH { return Err(err("Module depth exceeds 64")); }
        heights.push(height);
        edges.push(children);
    }

    if used_sources.len() != manifest.sources.len() || used_prepared.len() != manifest.prepared.len() {
        return Err(err("unreferenced Module definitions"));
    }
    let mut reached = BTreeSet::new();
    let mut pending = vec![edges.len() - 1];
    while let Some(i) = pending.pop() {
        if reached.insert(i) { pending.extend(&edges[i]); }
    }
    if reached.len() != edges.len() { return Err(err("unreachable Module instances")); }

    let vocabs = manifest.vocabs.into_iter().map(|v| {
        if v.entries.windows(2).any(|p| p[0].0 >= p[1].0) || v.exact_only.windows(2).any(|p| p[0] >= p[1])
            || v.exact_only.iter().any(|id| v.entries.binary_search_by_key(id, |e| e.0).is_ok()) {
            return Err(err("duplicate or unordered vocabulary IDs"));
        }
        let vocab = Vocab::new_with_exact_token_ids(v.entries, v.exact_only);
        if supplied.is_some_and(|v| !v.same_model_vocab(&vocab)) { return Err(err("Module exact events target a different vocabulary")); }
        Ok(supplied.cloned().unwrap_or(vocab))
    }).collect::<Result<Vec<_>>>()?;

    let sources = manifest.sources.into_iter().map(|s| decode_source(s, &vocabs).map(Arc::new)).collect::<Result<Vec<_>>>()?;
    let prepared = manifest.prepared.into_iter().map(|p| {
        prepared_artifact::load(&p.bytes, supplied).map(retain_local)
    }).collect::<Result<Vec<_>>>()?;

    let mut instances = Vec::<Module>::new();
    for node in manifest.instances {
        let mask_strategy = u8_to_engine(node.mask_strategy)?;
        let boundary_mask_strategy = u8_to_boundary(node.boundary_mask_strategy)?;
        let source = sources[node.source as usize].clone();
        let mut retained_map = BTreeMap::new();
        for (engine_u8, prep_id) in node.prepared {
            let engine = u8_to_engine(engine_u8)?;
            retained_map.insert(engine, prepared[prep_id as usize].clone());
        }
        let mut bindings = OrdMap::new();
        for (name, value) in node.bindings {
            let value = match value {
                Value::Module(i) => BoundValue::Module(instances[i as usize].clone()),
                Value::Token(id) => BoundValue::Token(Token::new(id)),
                Value::Exact(e) => BoundValue::ExactTokens(decode_exact(e, &vocabs)?),
            };
            bindings.insert(name, value);
        }
        let module = Module {
            node: Arc::new(Node {
                source,
                mask_strategy,
                boundary_mask_strategy,
                prepared: retained_map,
                bindings,
                certificate: Arc::new(OnceLock::new()),
            }),
        };
        if !module.node.prepared.is_empty() || !module.node.bindings.is_empty() {
            let declared = module.node.source.declarations().map_err(|e| err(e.to_string()))?;
            for (name, value) in &module.node.bindings {
                let kind = *declared.kinds.get(name).ok_or_else(|| err("Module binding has no declared slot"))?;
                check_binding_kind(name, kind, matches!(value, BoundValue::Module(_))).map_err(|e| err(e.to_string()))?;
            }
            for (&engine, p) in &module.node.prepared {
                let mut expected = declared.kinds.clone();
                let mut generated = BTreeSet::new();
                if matches!(module.node.source.source, GrammarSource::Lark(_)) {
                    for (name, _) in &module.node.source.lark_document().map_err(|e| err(e.to_string()))?.schemas {
                        if engine != MaskStrategy::TokenTrie {
                            expected.insert(name.clone(), ExternalKind::Grammar);
                            generated.insert(name.clone());
                        }
                    }
                }
                if p.slot_kinds.as_ref() != &expected || p.bindings.keys().cloned().collect::<BTreeSet<_>>() != generated {
                    return Err(err("retained Module interface/options mismatch"));
                }
                let native = p.inner.has_template_parser();
                if native && p.inner.table.is_present() { return Err(err("native Module artifact retains physical LR storage")); }
                if !native && !p.inner.table.is_present() { return Err(err("Module artifact has no supported parser representation")); }
                if engine != MaskStrategy::TokenTrie && !native { return Err(err("native Module artifact contains LR runtime")); }
                if engine == MaskStrategy::TokenTrie && !p.inner.uses_dynamic_runtime() {
                    return Err(err("FastBuild Module artifact contains a non-Dynamic local body"));
                }
            }
        }
        instances.push(module);
    }
    let root = instances.pop().ok_or_else(|| err("empty Module artifact"))?;
    root.check_limits().map_err(|e| err(e.to_string()))?;
    root.validate_targets(supplied).map_err(|e| err(e.to_string()))?;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(data: &[u8]) -> Manifest { bincode::deserialize(&data[16..]).unwrap() }
    fn write(m: &Manifest) -> Vec<u8> {
        let body = bincode::serialize(m).unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
        bytes.extend(body);
        bytes
    }
    fn fixture() -> Module {
        let mut m = Module::new(&Grammar::from_glrm("glrm 1; start s; extern grammar child; nt s = child;"));
        m.bind("child", Grammar::from_ebnf("start ::= \"a\"")).unwrap();
        m
    }
    #[test]
    fn corrupt_graphs_and_retained_option_claims_are_rejected_before_linking() {
        let mut module = fixture();
        let v = Vocab::new(vec![(0, b"a".to_vec())]);
        module.precompile(&v).unwrap();
        let bytes = module.save();
        let mut m = read(&bytes);
        m.instances.last_mut().unwrap().bindings[0].1 = Value::Module(1);
        assert!(Module::load(&write(&m), None).is_err());
        let mut m = read(&bytes);
        m.instances.last_mut().unwrap().mask_strategy = 99;
        assert!(Module::load(&write(&m), None).is_err());
        let mut m = read(&bytes);
        m.instances.last_mut().unwrap().source = 999;
        assert!(Module::load(&write(&m), None).is_err());
        let mut m = read(&bytes);
        let binding = m.instances.last_mut().unwrap().bindings.pop().unwrap();
        m.instances.last_mut().unwrap().bindings.push((binding.0.clone(), Value::Module(0)));
        m.instances.last_mut().unwrap().bindings.push((binding.0, Value::Module(0)));
        assert!(Module::load(&write(&m), None).is_err());
        let mut m = read(&bytes);
        m.prepared[0].bytes.truncate(20);
        assert!(Module::load(&write(&m), None).is_err());
    }
    #[test]
    fn exact_target_references_and_ranges_are_validated() {
        let v = Vocab::new_with_exact_token_ids(vec![(0, b"a".to_vec())], [7]);
        let mut module = Module::new(&Grammar::from_glrm("glrm 1; start s; extern token mark; nt s = mark;"));
        module.bind("mark", v.tokens([7]).unwrap()).unwrap();
        let bytes = module.save();
        let mut m = read(&bytes);
        if let Value::Exact(e) = &mut m.instances[0].bindings[0].1 { e.target = 99; }
        assert!(Module::load(&write(&m), None).is_err());
        let mut m = read(&bytes);
        if let Value::Exact(e) = &mut m.instances[0].bindings[0].1 { e.ranges = vec![(7, 7), (7, 7)]; }
        assert!(Module::load(&write(&m), None).is_err());
        let mut m = read(&bytes);
        m.vocabs[0].exact_only.push(0);
        assert!(Module::load(&write(&m), None).is_err());
    }
    #[test]
    fn tiny_manifests_cannot_claim_unbounded_vector_allocations() {
        for offset in [0, 8, 16, 24] {
            let mut body = vec![0u8; 32];
            body[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
            bytes.extend(body);
            let error = Module::load(&bytes, None).unwrap_err().to_string();
            assert!(error.contains("excessive artifact element count"), "{offset}: {error}");
        }
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(Module::load(&bytes, None).is_err());
    }
    #[test]
    fn relabelled_lr_as_native_and_immutable_local_bindings_are_rejected() {
        let v = Vocab::new(vec![(0, b"a".to_vec())]);
        let mut module = Module::new(&Grammar::from_ebnf("start ::= \"a\""));
        module.set_mask_strategy(MaskStrategy::ParserDwa).unwrap();
        module.precompile(&v).unwrap();
        let mut m = read(&module.save());
        let inner = crate::import::compile_dynamic_named_single_with_backend(
            Grammar::from_ebnf("start ::= \"a\"").resolve_dynamic_source(&v, false).unwrap(),
            &v,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
            false,
            crate::runtime::parser_backend::DynamicParserBackend::ordinary_dynamic(),
        ).unwrap();
        let _ = inner.late_bind_vocab.set(v.clone());
        let non_native = PreparedGrammar {
            inner: Arc::new(inner),
            bindings: OrdMap::new(),
            analysis: graph::empty_cache(),
            event_adapters: Arc::new(event_cache::ExactAdapterCache::default()),
            boundary_trigger: BoundaryTriggerDetail::None,
            slot_kinds: Arc::new(BTreeMap::new()),
            slot_spans: Arc::new(BTreeMap::new()),
        };
        m.prepared[0].bytes = non_native.save();
        let error = Module::load(&write(&m), None).unwrap_err().to_string();
        assert!(error.contains("native Module artifact contains LR"), "{error}");
        let mut m = read(&module.save());
        m.instances[0].source = 1;
        m.sources.push(Source::Ebnf("start ::= \"b\"".into()));
        assert!(Module::load(&write(&m), None).is_err(), "preparation must match its immutable source reference");
    }
}
