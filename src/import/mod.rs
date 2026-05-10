pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

use crate::compiler::compile::{compile_owned_profiled, compile_profile_enabled, emit_compile_profile_summary};
use crate::compiler::compile_owned;
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::grammar::named_simplify::simplify_named_grammar;
use crate::grammar::terminal_choice_promotion::promote_choice_terminals_exact;
use crate::runtime::Constraint;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;
type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;

const COMPILE_RESULT_CACHE_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompileResultCacheKey {
    source_kind_hash: u64,
    source_hash: u64,
    source_len: usize,
    vocab_entries_ptr: usize,
    vocab_len: usize,
    vocab_max_token_id: u32,
    eos_token_id: Option<u32>,
}

#[derive(Clone)]
struct CompileResultCacheEntry {
    key: CompileResultCacheKey,
    source_kind: String,
    source: Arc<str>,
    _vocab_entries: Arc<BTreeMap<u32, Vec<u8>>>,
    constraint: Constraint,
}

static COMPILE_RESULT_CACHE: OnceLock<Mutex<Vec<CompileResultCacheEntry>>> = OnceLock::new();

fn compile_result_cache_enabled() -> bool {
    !matches!(
        std::env::var("GLRMASK_COMPILE_RESULT_CACHE").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF")
    )
}

fn compile_result_cache_enabled_for_source_kind(source_kind: &str) -> bool {
    source_kind == "glrm" && compile_result_cache_enabled()
}

fn stable_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn compile_result_cache_key(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
) -> CompileResultCacheKey {
    CompileResultCacheKey {
        source_kind_hash: stable_hash(source_kind),
        source_hash: stable_hash(source),
        source_len: source.len(),
        vocab_entries_ptr: Arc::as_ptr(&vocab.entries) as usize,
        vocab_len: vocab.len(),
        vocab_max_token_id: vocab.max_token_id(),
        eos_token_id: vocab.eos_token_id,
    }
}

fn compile_result_cache_get(
    key: CompileResultCacheKey,
    source: &str,
    source_kind: &str,
) -> Option<Constraint> {
    let cache = COMPILE_RESULT_CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut cache = cache.lock().ok()?;
    let hit_index = cache.iter().position(|entry| {
        entry.key == key && entry.source_kind == source_kind && entry.source.as_ref() == source
    })?;
    let entry = cache.remove(hit_index);
    let constraint = entry.constraint.clone();
    cache.push(entry);
    Some(constraint)
}

fn compile_result_cache_insert(
    key: CompileResultCacheKey,
    source: &str,
    source_kind: &str,
    vocab: &crate::Vocab,
    constraint: &Constraint,
) {
    let cache = COMPILE_RESULT_CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let Ok(mut cache) = cache.lock() else {
        return;
    };
    if let Some(existing_index) = cache.iter().position(|entry| entry.key == key) {
        cache.remove(existing_index);
    }
    if cache.len() >= COMPILE_RESULT_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(CompileResultCacheEntry {
        key,
        source_kind: source_kind.to_owned(),
        source: Arc::<str>::from(source),
        _vocab_entries: Arc::clone(&vocab.entries),
        constraint: constraint.clone(),
    });
}

pub(crate) fn choice_or_single(mut options: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    if options.len() == 1 {
        options.pop().unwrap()
    } else {
        ast::GrammarExpr::Choice(options)
    }
}

pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    match items.len() {
        0 => ast::GrammarExpr::Sequence(Vec::new()),
        1 => items.pop().unwrap(),
        _ => ast::GrammarExpr::Sequence(items),
    }
}

fn lower_factored_named_grammar(
    source: &str,
    source_kind: &str,
    parse_named: NamedGrammarParser,
) -> crate::Result<GrammarDef> {
    let named = parse_named(source)?;
    let mut factored = factor_named_grammar(named);
    if source_kind == "json_schema" {
        if json_schema::simplify_grammar_enabled() {
            simplify_named_grammar(&mut factored);
        }
        if json_schema::promote_literal_choices_enabled() {
            promote_choice_terminals_exact(&mut factored, false);
        }
    }
    ast::lower(&factored)
}

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
    parse: NamedGrammarParser,
) -> crate::Result<Constraint> {
    let cache_key = if !compile_profile_enabled() && compile_result_cache_enabled_for_source_kind(source_kind) {
        let key = compile_result_cache_key(source, vocab, source_kind);
        if let Some(constraint) = compile_result_cache_get(key, source, source_kind) {
            return Ok(constraint);
        }
        Some(key)
    } else {
        None
    };

    if compile_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
    let constraint = compile_owned(grammar, vocab);
    if let Some(key) = cache_key {
        compile_result_cache_insert(key, source, source_kind, vocab, &constraint);
    }
    Ok(constraint)
}

fn parse_json_schema_to_named(schema_json: &str) -> crate::Result<ast::NamedGrammar> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    json_schema::schema_to_named_grammar(&schema)
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(ebnf, vocab, "ebnf", ebnf::parse_ebnf_to_named)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(lark, vocab, "lark", lark::parse_lark_to_named)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(schema, vocab, "json_schema", parse_json_schema_to_named)
    }

    /// Load a grammar from the GLRM format (see [`crate::grammar::glrm`]).
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(glrm, vocab, "glrm", crate::grammar::glrm::from_glrm)
    }
}
