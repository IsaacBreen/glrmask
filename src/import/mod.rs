pub use crate::grammar::ast as ast;
pub use glrmask_grammar::import::ebnf;
pub mod json_schema;
pub use glrmask_grammar::import::lark;

use std::collections::BTreeSet;

use crate::compiler::compile::{
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_profile_enabled,
    compile_top_profile_enabled,
    emit_compile_profile_summary,
};
use crate::compiler::pipeline::{
    compile_dynamic_owned_unfinalized_with_table_construction,
    compile_dynamic_owned_with_table_construction,
};
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::runtime::Constraint;
use crate::DynamicConstraint;

fn parse_ebnf_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(ebnf::parse_ebnf_to_named(source)?)
}

fn parse_lark_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(lark::parse_lark_to_named(source)?)
}

fn parse_glrm_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(crate::grammar::glrm::from_glrm(source)?)
}

fn prepare_json_schema_named(grammar: &mut ast::NamedGrammar) -> crate::Result<()> {
    Ok(json_schema::prepare_named_grammar(grammar)?)
}

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;
type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;
type NamedGrammarTransform = fn(&mut ast::NamedGrammar) -> crate::Result<()>;

const LARGE_IMPORT_SOURCE_BYTES: usize = 64 * 1024;
#[cfg(windows)]
const WINDOWS_LARGE_IMPORT_STACK_BYTES: usize = 64 * 1024 * 1024;

fn with_large_import_stack<T, F>(source_len: usize, compile: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    #[cfg(windows)]
    {
        if source_len >= LARGE_IMPORT_SOURCE_BYTES {
            return std::thread::scope(|scope| {
                let thread = std::thread::Builder::new()
                    .name("glrmask-grammar-compile".to_owned())
                    .stack_size(WINDOWS_LARGE_IMPORT_STACK_BYTES)
                    .spawn_scoped(scope, compile)
                    .expect("failed to spawn large-stack grammar compiler thread");
                match thread.join() {
                    Ok(result) => result,
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            });
        }
    }

    #[cfg(not(windows))]
    let _ = source_len;

    compile()
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

fn append_end_token_choice(grammar: &mut ast::NamedGrammar, end_token_ids: &[u32]) {
    let end_token_ids = end_token_ids.iter().copied().collect::<BTreeSet<_>>();
    if end_token_ids.is_empty() {
        return;
    }

    let original_start = grammar.start.clone();
    let base = "__glrmask_start_with_end_token";
    let mut generated_start = base.to_owned();
    let mut suffix = 2usize;
    while grammar.rules.iter().any(|rule| rule.name == generated_start) {
        generated_start = format!("{base}_{suffix}");
        suffix += 1;
    }

    let end = choice_or_single(
        end_token_ids
            .into_iter()
            .map(ast::GrammarExpr::SpecialToken)
            .collect(),
    );
    grammar.rules.push(ast::NamedRule {
        name: generated_start.clone(),
        expr: sequence_or_single(vec![ast::GrammarExpr::Ref(original_start), end]),
        is_terminal: false,
        is_internal: false,
    });
    grammar.start = generated_start;
}

fn emit_import_phase_start(name: &'static str) -> Option<std::time::Instant> {
    if !compile_profile_enabled() {
        return None;
    }

    eprintln!("[glrmask/profile][import-phase-start] name={}", name);
    Some(std::time::Instant::now())
}

fn emit_import_phase_end(name: &'static str, started_at: Option<std::time::Instant>) {
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][import-phase-end] name={} elapsed_ms={:.3}",
            name,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

fn lower_factored_named_grammar(
    source: &str,
    parse_named: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<GrammarDef> {
    let lower_started_at = emit_import_phase_start("lower_factored_named_grammar");
    let parse_named_started_at = emit_import_phase_start("parse_named");
    let named = parse_named(source)?;
    emit_import_phase_end("parse_named", parse_named_started_at);

    let factor_started_at = emit_import_phase_start("factor_named_grammar");
    let mut factored = factor_named_grammar(named);
    emit_import_phase_end("factor_named_grammar", factor_started_at);

    if let Some(transform) = transform {
        let transform_started_at = emit_import_phase_start("transform_named_grammar");
        transform(&mut factored)?;
        emit_import_phase_end("transform_named_grammar", transform_started_at);
    }
    append_end_token_choice(&mut factored, end_token_ids);

    let ast_lower_started_at = emit_import_phase_start("ast_lower");
    let grammar = ast::lower(&factored);
    emit_import_phase_end("ast_lower", ast_lower_started_at);
    emit_import_phase_end("lower_factored_named_grammar", lower_started_at);
    Ok(grammar?)
}

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Constraint> {
    let compile_from_source_started_at = emit_import_phase_start("compile_from_source");
    if compile_profile_enabled() || compile_top_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = lower_factored_named_grammar(source, parse, transform, end_token_ids)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = crate::error::catch_internal_invariant(|| {
            compile_owned_profiled_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )
        })?;
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        emit_import_phase_end("compile_from_source", compile_from_source_started_at);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, parse, transform, end_token_ids)?;
    let constraint = crate::error::catch_internal_invariant(|| {
        compile_owned_with_table_construction(grammar, vocab, default_table_construction)
    })?;
    emit_import_phase_end("compile_from_source", compile_from_source_started_at);
    Ok(constraint)
}

fn dynamic_named_alternatives(
    source: &str,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<ast::NamedGrammar>> {
    let mut factored = factor_named_grammar(parse(source)?);
    if let Some(transform) = transform {
        transform(&mut factored)?;
    }

    let start_index = factored
        .rules
        .iter()
        .position(|rule| !rule.is_terminal && rule.name == factored.start);
    let options = start_index.and_then(|index| match &factored.rules[index].expr {
        ast::GrammarExpr::Choice(options) if options.len() > 1 => Some(options.clone()),
        _ => None,
    });
    let has_embedded_region = factored.rules.iter().any(|rule| {
        !rule.is_terminal
            && matches!(
                &rule.expr,
                ast::GrammarExpr::ExprNFA(expr_nfa)
                    if expr_nfa.prefer_direct_nfa_emission
                        && !expr_nfa.complete_parser_language
            )
    });

    if !has_embedded_region || options.is_none() {
        append_end_token_choice(&mut factored, end_token_ids);
        return Ok(vec![factored]);
    }

    let start_index = start_index.expect("start choice index was resolved");
    let mut alternatives = Vec::new();
    for option in options.expect("start choice options were resolved") {
        let mut alternative = factored.clone();
        alternative.rules[start_index].expr = option.clone();
        if let ast::GrammarExpr::Ref(region_name) = &option
            && let Some(region_rule) = alternative.rules.iter_mut().find(|rule| {
                !rule.is_terminal && rule.name == *region_name
            })
            && let ast::GrammarExpr::ExprNFA(expr_nfa) = &mut region_rule.expr
            && expr_nfa.prefer_direct_nfa_emission
        {
            expr_nfa.complete_parser_language = true;
            alternative.start = region_name.clone();
        }
        crate::grammar::right_linear::retain_reachable_rules(&mut alternative);
        append_end_token_choice(&mut alternative, end_token_ids);
        alternatives.push(alternative);
    }
    Ok(alternatives)
}

fn compile_dynamic_from_source(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<DynamicConstraint> {
    let alternatives = dynamic_named_alternatives(source, parse, transform, end_token_ids)?;
    let mut compiled = Vec::with_capacity(alternatives.len());
    for alternative in alternatives {
        let grammar = ast::lower(&alternative)?;
        compiled.push(compile_dynamic_owned_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        ));
    }
    Ok(DynamicConstraint::from_alternatives(compiled))
}

fn compile_dynamic_serialized_from_source_profiled(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<(Vec<u8>, u64, u64)> {
    let wall_started = std::time::Instant::now();
    let profile = compile_profile_enabled() || compile_top_profile_enabled();
    let total_started = profile.then(std::time::Instant::now);
    let import_started = profile.then(std::time::Instant::now);
    let alternatives = dynamic_named_alternatives(source, parse, transform, end_token_ids)?;
    let import_ms = import_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compile_started = profile.then(std::time::Instant::now);
    let mut compiled = Vec::with_capacity(alternatives.len());
    let mut lower_ms = 0.0;
    for alternative in alternatives {
        let lower_started = profile.then(std::time::Instant::now);
        let grammar = ast::lower(&alternative)?;
        lower_ms += lower_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        compiled.push(compile_dynamic_owned_unfinalized_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        ));
    }
    let compile_ms = compile_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compile_ns = wall_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let serialize_wall_started = std::time::Instant::now();
    let serialize_started = profile.then(std::time::Instant::now);
    let bytes = DynamicConstraint::from_alternatives(compiled).into_saved();
    let serialize_ns = serialize_wall_started
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let serialize_ms = serialize_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][dynamic_serialized_source] import_ms={:.3} lower_ms={:.3} compile_with_lower_ms={:.3} serialize_ms={:.3} bytes={} total_ms={:.3}",
            import_ms,
            lower_ms,
            compile_ms,
            serialize_ms,
            bytes.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((bytes, compile_ns, serialize_ns))
}

fn compile_dynamic_serialized_from_source(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<u8>> {
    compile_dynamic_serialized_from_source_profiled(
        source,
        vocab,
        default_table_construction,
        parse,
        transform,
        end_token_ids,
    )
    .map(|(bytes, _, _)| bytes)
}

/// Profiling-only entry point: runs the JSON-schema import pipeline
/// (parse → factor → AST lower) without the downstream compile. Hidden from the
/// public API; used by `examples/profile_glr.rs` to isolate import timings.
#[doc(hidden)]
pub fn __profile_json_schema_import(schema_json: &str) -> crate::Result<()> {
    let grammar = lower_factored_named_grammar(
        schema_json,
        parse_json_schema_to_named,
        Some(prepare_json_schema_named),
        &[],
    )?;
    std::hint::black_box(&grammar);
    Ok(())
}

fn parse_json_schema_to_named(schema_json: &str) -> crate::Result<ast::NamedGrammar> {
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);

    let schema_to_named_started_at = emit_import_phase_start("schema_to_named_grammar");
    let named = json_schema::schema_to_named_grammar(&schema);
    emit_import_phase_end("schema_to_named_grammar", schema_to_named_started_at);
    Ok(named?)
}

impl Constraint {
    /// Compile an EBNF grammar for `vocab`.
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_ebnf_with_end_tokens(ebnf, vocab, &[])
    }

    /// Compile an EBNF grammar and declare model end-token IDs.
    pub fn from_ebnf_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(ebnf.len(), || {
            compile_from_source(
                ebnf,
                vocab,
                "ebnf",
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a Lark grammar for `vocab`.
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_lark_with_end_tokens(lark, vocab, &[])
    }

    /// Compile a Lark grammar and declare model end-token IDs.
    pub fn from_lark_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(lark.len(), || {
            compile_from_source(
                lark,
                vocab,
                "lark",
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a JSON Schema for `vocab`.
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    /// Compile a JSON Schema and declare model end-token IDs.
    pub fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            crate::compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
                compile_from_source(
                    schema,
                    vocab,
                    "json_schema",
                    GlrTableConstruction::LegacyRowBisim,
                    parse_json_schema_to_named,
                    Some(prepare_json_schema_named),
                    end_token_ids,
                )
            })
        })
    }

    /// Compile a grammar in GLRMask's native GLRM format.
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_end_tokens(glrm, vocab, &[])
    }

    /// Compile a GLRM grammar and declare model end-token IDs.
    pub fn from_glrm_grammar_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            compile_from_source(
                glrm,
                vocab,
                "glrm",
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
            )
        })
    }
}

impl DynamicConstraint {
    #[doc(hidden)]
    pub fn compile_ebnf_serialized_profiled_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    #[doc(hidden)]
    pub fn compile_lark_serialized_profiled_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    #[doc(hidden)]
    pub fn compile_json_schema_serialized_profiled_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named,
                Some(prepare_json_schema_named),
                end_token_ids,
            )
        })
    }

    #[doc(hidden)]
    pub fn compile_glrm_serialized_profiled_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile EBNF directly to a serialized dynamic artifact without building
    /// runtime-only caches in the producing process.
    #[doc(hidden)]
    pub fn compile_ebnf_serialized_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_serialized_from_source(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile Lark directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub fn compile_lark_serialized_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_serialized_from_source(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile JSON Schema directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub fn compile_json_schema_serialized_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_serialized_from_source(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named,
                Some(prepare_json_schema_named),
                end_token_ids,
            )
        })
    }

    /// Compile GLRM directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub fn compile_glrm_serialized_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_serialized_from_source(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile an EBNF grammar with reduced compilation latency.
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_ebnf_with_end_tokens(ebnf, vocab, &[])
    }

    /// Compile an EBNF grammar with reduced latency and model end-token IDs.
    pub fn from_ebnf_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_from_source(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a Lark grammar with reduced compilation latency.
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_lark_with_end_tokens(lark, vocab, &[])
    }

    /// Compile a Lark grammar with reduced latency and model end-token IDs.
    pub fn from_lark_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_from_source(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a JSON Schema with reduced compilation latency.
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    /// Compile a JSON Schema with reduced latency and model end-token IDs.
    pub fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_from_source(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named,
                Some(prepare_json_schema_named),
                end_token_ids,
            )
        })
    }

    /// Compile a GLRM grammar with reduced compilation latency.
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_end_tokens(glrm, vocab, &[])
    }

    /// Compile a GLRM grammar with reduced latency and model end-token IDs.
    pub fn from_glrm_grammar_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_from_source(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
            )
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::table::{AdmissionPolicy, GlrTableConstruction};
    use crate::Vocab;

    #[cfg(windows)]
    #[test]
    fn large_imports_use_dedicated_windows_stack() {
        let thread_name = with_large_import_stack(LARGE_IMPORT_SOURCE_BYTES, || {
            std::thread::current().name().map(str::to_owned)
        });
        assert_eq!(thread_name.as_deref(), Some("glrmask-grammar-compile"));
    }

    fn vocab(entries: &[&str]) -> Vocab {
        Vocab::new(
            entries
                .iter()
                .enumerate()
                .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
                .collect())
    }

    #[test]
    fn json_schema_import_uses_legacy_row_bisim_table_by_default() {
        let constraint = Constraint::from_json_schema(
            r#"{"type":"string"}"#,
            &vocab(&["\"", "a", "\"a\""]),
        )
        .unwrap();

        assert_eq!(constraint.table.construction, GlrTableConstruction::LegacyRowBisim);
        assert_eq!(constraint.table.admission_policy, AdmissionPolicy::RowPresenceExact);
    }

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        mask.get(token_id as usize / 32)
            .is_some_and(|word| word & (1u32 << (token_id % 32)) != 0)
    }

    #[test]
    fn json_schema_end_tokens_are_exact_parser_terminals() {
        let vocab = vocab(&["\"", "a", "\"a\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[101, 100, 101],
        )
        .unwrap();
        assert_eq!(constraint.mask_len(), 4);

        let mut state = constraint.start();
        assert!(!token_allowed(&state.mask(), 100));
        assert!(!token_allowed(&state.mask(), 101));
        state.commit_token(2).unwrap();
        assert!(!state.is_complete());
        let mask = state.mask();
        assert!(token_allowed(&mask, 100));
        assert!(token_allowed(&mask, 101));
        assert_eq!(state.forced(), Vec::<u32>::new());
        state.commit_token(100).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn json_schema_single_end_token_is_forced() {
        let vocab = vocab(&["\"", "a", "\"a\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();

        let mut state = constraint.start();
        state.commit_token(2).unwrap();
        assert_eq!(state.forced(), vec![100]);
        state.commit_token(100).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn json_schema_end_token_can_also_have_byte_semantics() {
        let vocab = Vocab::new(vec![(100, b"\"a\"".to_vec())]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();

        let mut state = constraint.start();
        assert_eq!(state.forced(), vec![100, 100]);
        state.commit_token(100).unwrap();
        assert!(!state.is_complete());
        assert_eq!(state.forced(), vec![100]);
        state.commit_token(100).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn caller_sized_masks_zero_unknown_trailing_tokens() {
        let vocab = vocab(&["\"a\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();
        let state = constraint.start();
        let mut oversized = vec![u32::MAX; constraint.mask_len() + 3];
        state.fill_mask(&mut oversized);
        assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));
        oversized.fill(u32::MAX);
        state.fill_mask(&mut oversized);
        assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));

        let dynamic = DynamicConstraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();
        let mut dynamic_mask = vec![u32::MAX; dynamic.mask_len() + 3];
        let dynamic_state = dynamic.start();
        dynamic_state.fill_mask(&mut dynamic_mask);
        assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
        dynamic_mask.fill(u32::MAX);
        dynamic_state.fill_mask(&mut dynamic_mask);
        assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
    }

    #[test]
    fn glrm_import_uses_core_merged_table_by_default() {
        let constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= 'a' ;\nnt start ::= A ;\n",
            &vocab(&["a"]),
        )
        .unwrap();

        assert_eq!(
            constraint.table.construction,
            GlrTableConstruction::ExperimentalCoreMerged
        );
        assert_eq!(constraint.table.admission_policy, AdmissionPolicy::ExactSimulation);
    }

    #[test]
    fn ebnf_import_uses_core_merged_table_by_default() {
        let constraint = Constraint::from_ebnf("start ::= 'a'", &vocab(&["a"])).unwrap();

        assert_eq!(
            constraint.table.construction,
            GlrTableConstruction::ExperimentalCoreMerged
        );
        assert_eq!(constraint.table.admission_policy, AdmissionPolicy::ExactSimulation);
    }
}
