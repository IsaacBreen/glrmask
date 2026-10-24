//! Versioned postorder DAG. Sources and retained compiled definitions are interned separately.
use super::*;
use bincode::Options;
use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
const MAGIC: &[u8; 8] = b"GLRASM03";
const MAX_BYTES: u64 = 1 << 30;

#[derive(Serialize, Deserialize)]
struct Exact {
    #[serde(deserialize_with = "artifact_limits::items")]
    ranges: Vec<(u32, u32)>,
}

#[derive(Serialize, Deserialize)]
enum Source {
    Ebnf(String),
    Glrm(String),
    Json(String),
    Lark(String),
    Exact(Exact),
}

#[derive(Serialize, Deserialize)]
enum Value {
    Module(u32),
    Exact(Exact),
}

#[derive(Serialize, Deserialize)]
struct Instance {
    source: u32,
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
    #[serde(default)]
    token_set: Option<Vec<u32>>,
    bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    #[serde(deserialize_with = "artifact_limits::nodes")]
    sources: Vec<Source>,
    #[serde(deserialize_with = "artifact_limits::nodes")]
    prepared: Vec<Preparation>,
    #[serde(deserialize_with = "artifact_limits::nodes")]
    instances: Vec<Instance>,
}

fn err(message: impl Into<String>) -> Error { Error::Serialization(message.into()) }

fn u8_to_engine(n: u8) -> Result<Engine> {
    match n {
        0 => Ok(Engine::Auto),
        1 => Ok(Engine::ParserDwa),
        2 => Ok(Engine::TokenTrie),
        3 => Ok(Engine::MergedTokenTrie),
        _ => Err(err("invalid mask strategy tag")),
    }
}

fn exact(tokens: &TokenSet) -> Exact {
    Exact { ranges: tokens.range_pairs().to_vec() }
}

fn source(grammar: &Grammar) -> Source {
    if let Some(t) = &grammar.exact_value { return Source::Exact(exact(t)); }
    match &grammar.source {
        GrammarSource::Ebnf(s) => Source::Ebnf(s.to_string()),
        GrammarSource::Glrm(s) => Source::Glrm(s.to_string()),
        GrammarSource::JsonSchema(s) => Source::Json(s.to_string()),
        GrammarSource::Lark(s) => Source::Lark(s.to_string()),
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

    for module in modules {
        let node = &module.node;
        let source_key = Arc::as_ptr(&node.source.parsed_component) as usize;
        let source_id = *source_ids.entry(source_key).or_insert_with(|| {
            let id = sources.len() as u32;
            sources.push(source(&node.source));
            id
        });
        let mut preps = Vec::new();
        for (&engine, p) in &node.prepared {
            let engine_u8 = engine;
            let prep_key = Arc::as_ptr(p) as *const () as usize;
            let prep_id = *prepared_ids.entry(prep_key).or_insert_with(|| {
                let id = prepared.len() as u32;
                prepared.push(Preparation {
                    source: source_id,
                    engine: engine_u8,
                    boundary: p.prepared().boundary_trigger as u8,
                    token_set: match &p.prepared().inner.boundary_trigger {
                        crate::runtime::BoundaryTrigger::TokenSet(tokens) => Some(tokens.to_vec()),
                        _ => None,
                    },
                    bytes: p.prepared().save(),
                });
                id
            });
            preps.push((engine_u8, prep_id));
        }
        let bindings = node.bindings.iter().map(|(name, value)| {
            let value = match value {
                BoundValue::Module(m) => Value::Module(instance_ids[&(Arc::as_ptr(&m.node) as usize)]),
                BoundValue::TokenSet(t) => Value::Exact(exact(t)),
            };
            (name.clone(), value)
        }).collect();
        instance_ids.insert(Arc::as_ptr(node) as usize, instances.len() as u32);
        instances.push(Instance {
            source: source_id,
            prepared: preps,
            bindings,
        });
    }
    let manifest = Manifest {
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

fn decode_exact(value: Exact) -> Result<TokenSet> {
    let mut previous = None;
    for &(a, b) in &value.ranges {
        if a > b || previous.is_some_and(|p| u64::from(p) + 1 >= u64::from(a)) { return Err(err("unnormalized token-set ranges")); }
        previous = Some(b);
    }
    Ok(TokenSet::from_ranges(value.ranges.iter().map(|&(a, b)| a..=b)))
}

fn decode_source(value: Source) -> Result<Grammar> {
    let source = match value {
        Source::Ebnf(s) => GrammarSource::Ebnf(Arc::from(s)),
        Source::Glrm(s) => GrammarSource::Glrm(Arc::from(s)),
        Source::Json(s) => GrammarSource::JsonSchema(Arc::from(s)),
        Source::Lark(text) => return Ok(Grammar::new(GrammarSource::Lark(Arc::from(text)))),
        Source::Exact(e) => return Ok(Grammar::from_token_set(decode_exact(e)?)),
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
        || manifest.sources.len() > graph::MAX_NODES || manifest.prepared.len() > graph::MAX_NODES {
        return Err(err("invalid or excessive Module graph size"));
    }

    let mut heights = Vec::new();
    let mut edges = Vec::new();
    let mut used_sources = BTreeSet::new();
    let mut used_prepared = BTreeSet::new();
    let mut ports = 0usize;

    for (i, node) in manifest.instances.iter().enumerate() {
        if node.source as usize >= manifest.sources.len() {
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

    let sources = manifest.sources.into_iter().map(|s| decode_source(s).map(Arc::new)).collect::<Result<Vec<_>>>()?;
    let prepared = manifest.prepared.into_iter().map(|p| {
        let mut prepared = prepared_artifact::load(&p.bytes, supplied)?;
        if let Some(mut tokens) = p.token_set {
            tokens.sort_unstable();
            tokens.dedup();
            Arc::make_mut(&mut prepared.inner).boundary_trigger =
                crate::runtime::BoundaryTrigger::TokenSet(Arc::from(tokens.into_boxed_slice()));
        }
        Ok(retain_local(prepared))
    }).collect::<Result<Vec<_>>>()?;

    let mut instances = Vec::<Module>::new();
    for node in manifest.instances {
        let source = sources[node.source as usize].clone();
        let mut retained_map = BTreeMap::new();
        for (engine_u8, prep_id) in node.prepared {
            let engine = u8_to_engine(engine_u8)?;
            retained_map.insert(engine.key(), prepared[prep_id as usize].clone());
        }
        let mut bindings = OrdMap::new();
        for (name, value) in node.bindings {
            let value = match value {
                Value::Module(i) => BoundValue::Module(instances[i as usize].clone()),
                Value::Exact(e) => BoundValue::TokenSet(decode_exact(e)?),
            };
            bindings.insert(name, value);
        }
        let module = Module {
            node: Arc::new(Node {
                source,
                prepared: retained_map,
                bindings,
                certificate: Arc::new(OnceLock::new()),
                #[cfg(test)]
                force_boundary_dwa_decline: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                #[cfg(test)]
                force_parser_dwa_decline: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
                        if engine != Engine::TokenTrie.key() {
                            expected.insert(name.clone(), ExternKind::Grammar);
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
                if engine != Engine::TokenTrie.key() && !native { return Err(err("corrupt module: a precompiled part has an LR table")); }
                if engine == Engine::TokenTrie.key() && !p.inner.uses_dynamic_runtime() {
                    return Err(err("corrupt module: a TokenTrie part has the wrong runtime"));
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
        let mut m = Module::new(&Grammar::from_glrm("glrm 1; start s; extern grammar child; nt s = child;").unwrap());
        m.bind("child", Grammar::from_ebnf("start ::= \"a\"").unwrap()).unwrap();
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
        assert!(Module::load(&write(&m)).is_err());
        let mut m = read(&bytes);
        m.instances.last_mut().unwrap().prepared[0].1 = 99;
        assert!(Module::load(&write(&m)).is_err());
        let mut m = read(&bytes);
        m.instances.last_mut().unwrap().source = 999;
        assert!(Module::load(&write(&m)).is_err());
        let mut m = read(&bytes);
        let binding = m.instances.last_mut().unwrap().bindings.pop().unwrap();
        m.instances.last_mut().unwrap().bindings.push((binding.0.clone(), Value::Module(0)));
        m.instances.last_mut().unwrap().bindings.push((binding.0, Value::Module(0)));
        assert!(Module::load(&write(&m)).is_err());
        let mut m = read(&bytes);
        m.prepared[0].bytes.truncate(20);
        assert!(Module::load(&write(&m)).is_err());
    }
    #[test]
    fn token_set_ranges_are_validated() {
        let v = Vocab::new(vec![(0, b"a".to_vec())]).with_special_tokens([7]);
        let mut module = Module::new(&Grammar::from_glrm("glrm 1; start s; extern token mark; nt s = mark;").unwrap());
        module.bind("mark", TokenSet::from(7)).unwrap();
        let bytes = module.save();
        let mut m = read(&bytes);
        if let Value::Exact(e) = &mut m.instances[0].bindings[0].1 { e.ranges = vec![(7, 7), (7, 7)]; }
        assert!(Module::load(&write(&m)).is_err());

    }
    #[test]
    fn tiny_manifests_cannot_claim_unbounded_vector_allocations() {
        for offset in [0, 8, 16, 24] {
            let mut body = vec![0u8; 32];
            body[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
            bytes.extend(body);
            let error = Module::load(&bytes).unwrap_err().to_string();
            assert!(
                error.contains("excessive artifact element count")
                    || error.contains("Slice had bytes remaining after deserialization"),
                "{offset}: {error}",
            );
        }
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(Module::load(&bytes).is_err());
    }
    #[test]
    fn corrupt_key_2_body_is_rejected() {
        let v = Vocab::new(vec![(0, b"a".to_vec())]);
        let mut shell = Module::new(&Grammar::from_ebnf("start ::= \"a\"").unwrap());
        shell
            .precompile_with(
                &v,
                CompileOptions::default()
                    .mask_strategy(MaskStrategy::TokenTrie)
                    .parser_backend(ParserBackend::TemplateDfa),
            )
            .unwrap();
        let mut static_part = Module::new(&Grammar::from_ebnf("start ::= \"a\"").unwrap());
        static_part
            .precompile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa))
            .unwrap();
        let static_bytes = read(&static_part.save()).prepared[0].bytes.clone();
        let mut manifest = read(&shell.save());
        assert_eq!(manifest.prepared[0].engine, 2);
        manifest.prepared[0].bytes = static_bytes;
        let error = Module::load(&write(&manifest)).unwrap_err().to_string();
        assert!(
            error.contains("corrupt module: a TokenTrie part has the wrong runtime"),
            "{error}",
        );
    }

    #[test]
    fn relabelled_lr_as_native_and_immutable_local_bindings_are_rejected() {
        let v = Vocab::new(vec![(0, b"a".to_vec())]);
        let mut module = Module::new(&Grammar::from_ebnf("start ::= \"a\"").unwrap());
        module
            .precompile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa))
            .unwrap();
        let mut m = read(&module.save());
        let inner = crate::import::compile_dynamic_named_single_with_backend(
            Grammar::from_ebnf("start ::= \"a\"").unwrap().resolve_dynamic_source(&v, false).unwrap(),
            &v,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
            false,
            crate::runtime::parser_backend::DynamicParserBackend::LrTable,
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
            retained_artifact: None,
            retained_vocab_fingerprint: None,
        };
        m.prepared[0].bytes = non_native.save();
        let error = Module::load(&write(&m)).unwrap_err().to_string();
        assert!(error.contains("corrupt module: a precompiled part has an LR table"), "{error}");
        let mut m = read(&module.save());
        m.instances[0].source = 1;
        m.sources.push(Source::Ebnf("start ::= \"b\"".into()));
        assert!(Module::load(&write(&m)).is_err(), "preparation must match its immutable source reference");
    }
}
