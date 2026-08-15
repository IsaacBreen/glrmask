pub use crate::grammar::ast as ast;
pub(crate) use glrmask_grammar::__private::import::ebnf;
pub mod json_schema;
pub(crate) use glrmask_grammar::__private::import::lark;

use std::collections::{BTreeMap, BTreeSet};

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
        let (mut constraint, profile) = crate::error::catch_internal_invariant(|| {
            compile_owned_profiled_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )
        })?;
        constraint.table.set_embedded_end_token_ids(end_token_ids);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        emit_import_phase_end("compile_from_source", compile_from_source_started_at);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, parse, transform, end_token_ids)?;
    let mut constraint = crate::error::catch_internal_invariant(|| {
        compile_owned_with_table_construction(grammar, vocab, default_table_construction)
    })?;
    constraint.table.set_embedded_end_token_ids(end_token_ids);
    emit_import_phase_end("compile_from_source", compile_from_source_started_at);
    Ok(constraint)
}

fn compile_from_named_grammar(
    named: ast::NamedGrammar,
    vocab: &crate::Vocab,
    source_kind: &str,
    default_table_construction: GlrTableConstruction,
    end_token_ids: &[u32],
) -> crate::Result<Constraint> {
    let import_started_at = std::time::Instant::now();
    let mut factored = factor_named_grammar(named);
    append_end_token_choice(&mut factored, end_token_ids);
    let grammar = ast::lower(&factored)?;
    let import_ms = import_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() || compile_top_profile_enabled() {
        let (mut constraint, profile) = crate::error::catch_internal_invariant(|| {
            compile_owned_profiled_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )
        })?;
        constraint.table.set_embedded_end_token_ids(end_token_ids);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let mut constraint = crate::error::catch_internal_invariant(|| {
        compile_owned_with_table_construction(grammar, vocab, default_table_construction)
    })?;
    constraint.table.set_embedded_end_token_ids(end_token_ids);
    Ok(constraint)
}

fn first_external_placeholder_token_id(vocab: &crate::Vocab) -> crate::Result<u32> {
    if vocab.is_empty() {
        return Ok(0);
    }
    if let Some(next) = vocab.max_token_id().checked_add(1) {
        return Ok(next);
    }

    // Only the pathological `u32::MAX` case needs a vocabulary scan. Normal
    // dense model vocabularies stay on the constant-time max+1 path.
    let mut expected = 0u32;
    for (token_id, _) in vocab.iter() {
        if token_id != expected {
            return Ok(expected);
        }
        expected = expected.checked_add(1).ok_or_else(|| {
            crate::GlrMaskError::Compilation(
                "every u32 token ID is occupied; no external-subgrammar placeholder is available"
                    .to_string(),
            )
        })?;
    }
    Ok(expected)
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

    /// Compile GLRM containing typed `extern g name;` declarations and bind
    /// each declaration to an already-compiled child constraint.
    ///
    /// Binding names are the source names of top-level externals. Externals
    /// nested inside inline subgrammars use qualified names such as
    /// `outer::leaf`. Hidden non-vocabulary linker-control IDs are allocated
    /// automatically; callers never need to manufacture `@token(...)`
    /// sentinels. Missing, duplicate, and unknown bindings are rejected before
    /// compilation or linking.
    pub fn from_glrm_grammar_with_subgrammars(
        glrm: &str,
        children: &[(&str, &Constraint)],
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_subgrammars_and_end_tokens(glrm, children, vocab, &[])
    }

    /// Compile GLRM with typed external subgrammars and model end-token IDs.
    pub fn from_glrm_grammar_with_subgrammars_and_end_tokens(
        glrm: &str,
        children: &[(&str, &Constraint)],
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            let first_placeholder_token_id = first_external_placeholder_token_id(vocab)?;
            let parsed = crate::grammar::glrm::from_glrm_with_external_subgrammars(
                glrm,
                first_placeholder_token_id,
                end_token_ids.iter().copied().chain(
                    children
                        .iter()
                        .flat_map(|(_, child)| {
                            child
                                .special_token_terminals
                                .iter()
                                .map(|special| special.token_id)
                        }),
                ),
            )?;

            let mut children_by_name = BTreeMap::<&str, &Constraint>::new();
            for &(binding_name, child) in children {
                if children_by_name.insert(binding_name, child).is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "external subgrammar binding {binding_name:?} was supplied more than once",
                    )));
                }
            }

            let mut external_bindings = Vec::with_capacity(parsed.placeholders.len());
            for placeholder in &parsed.placeholders {
                let child = children_by_name
                    .remove(placeholder.binding_name.as_str())
                    .ok_or_else(|| {
                        crate::GlrMaskError::Compilation(format!(
                            "GLRM declares external subgrammar {:?}, but no compiled child was supplied",
                            placeholder.binding_name,
                        ))
                    })?;
                external_bindings.push((placeholder.token_id, placeholder.binding_name.as_str(), child));
            }
            if let Some((&unknown, _)) = children_by_name.first_key_value() {
                return Err(crate::GlrMaskError::Compilation(format!(
                    "compiled child was supplied for unknown external subgrammar {unknown:?}",
                )));
            }

            let parent = compile_from_named_grammar(
                parsed.grammar,
                vocab,
                "glrm",
                GlrTableConstruction::ExperimentalCoreMerged,
                end_token_ids,
            )?;
            if external_bindings.is_empty() {
                return Ok(parent);
            }

            let mut composition_inputs = Vec::with_capacity(external_bindings.len());
            for (placeholder_token_id, binding_name, child) in external_bindings {
                let mut matching_terminals = parent
                    .special_token_terminals
                    .iter()
                    .filter(|special| special.token_id == placeholder_token_id)
                    .map(|special| special.terminal_id);
                let placeholder_terminal = matching_terminals.next().ok_or_else(|| {
                    crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} lost its hidden linker terminal",
                    ))
                })?;
                if matching_terminals.next().is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} has multiple hidden linker terminals",
                    )));
                }
                composition_inputs.push(
                    crate::compiler::constraint_compose::CompiledSubgrammarInput {
                        placeholder_terminal,
                        constraint: child,
                    },
                );
            }
            crate::compiler::constraint_compose::compose_constraints_owned_parent(
                parent,
                &composition_inputs,
                vocab,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)
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
    fn glrm_import_merges_partially_mapped_terminal_families() {
        let mut entries = (0u32..=255)
            .map(|byte| (byte, vec![byte as u8]))
            .collect::<Vec<_>>();
        entries.extend([
            (256, b"{\"value\": ".to_vec()),
            (257, b"left".to_vec()),
            (258, b"right".to_vec()),
            (259, b"}".to_vec()),
        ]);
        let vocab = Vocab::new(entries);

        Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" "\"value\": " (LEFT | RIGHT) "}";
            "#,
            &vocab,
        )
        .expect("partial terminal-family maps should merge without indexing the unmapped sentinel");
    }

    #[test]
    fn glrm_uniform_subgrammar_ignore_uses_global_terminal_dwa_path() {
        let vocab = vocab(&[" ", "<", ">", "a", "b"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    ignore INNER_WS;
                    t INNER_WS ::= " "+;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(
            constraint.ignore_terminal.is_some(),
            "uniform scoped ignore should remain a global ignore terminal",
        );
        let mut state = constraint.start();
        state.commit_bytes(b"  < a   b >  ").unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn glrm_nested_uniform_subgrammar_ignore_uses_global_terminal_dwa_path() {
        let vocab = vocab(&[" ", "<", ">", "[", "]", "x"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore ROOT_WS;
                t ROOT_WS ::= " "+;

                g middle ::= {
                    start wrapped;
                    ignore MIDDLE_WS;
                    t MIDDLE_WS ::= " "+;

                    g leaf ::= {
                        start value;
                        ignore LEAF_WS;
                        t LEAF_WS ::= " "+;
                        nt value ::= "x";
                    };

                    nt wrapped ::= "[" leaf "]";
                };

                nt document ::= "<" middle ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(constraint.ignore_terminal.is_some());
        let mut state = constraint.start();
        state.commit_bytes(b" < [ x ] > ").unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn glrm_mixed_subgrammar_ignore_keeps_scoped_grammar_lowering() {
        let vocab = vocab(&[" ", "\t", "<", ">", "a", "b"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    ignore INNER_WS;
                    t INNER_WS ::= "\t"+;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(
            constraint.ignore_terminal.is_none(),
            "different scoped ignores must retain explicit scope-local lowering",
        );
        let mut state = constraint.start();
        state.commit_bytes(b" <\ta\t\tb\t> ").unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn glrm_child_without_ignore_keeps_scoped_grammar_lowering() {
        let vocab = vocab(&[" ", "<", ">", "a", "b"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(constraint.ignore_terminal.is_none());
        let mut state = constraint.start();
        state.commit_bytes(b" <ab> ").unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn glrm_external_subgrammar_api_matches_inline_grammar() {
        let vocab = vocab(&["X", "a", "b", "!", "Xa", "ab!", "Xab!"]);
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                extern g payload;
                nt document ::= "X" payload "!";
            "#,
            &[("payload", &child)],
            &vocab,
        )
        .unwrap();
        let inline = Constraint::from_glrm_grammar(
            r#"
                start document;
                g payload ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "X" payload "!";
            "#,
            &vocab,
        )
        .unwrap();

        for sequence in [vec![6], vec![0, 5], vec![4, 2, 3]] {
            let mut actual = composed.start();
            let mut expected = inline.start();
            for token_id in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token_id).unwrap();
                expected.commit_token(token_id).unwrap();
            }
            assert_eq!(actual.is_complete(), expected.is_complete());
            assert!(actual.is_complete());
        }
    }

    #[test]
    fn glrm_external_subgrammars_support_adjacent_reuse() {
        let vocab = vocab(&["X", "a", "!", "Xa", "aa!"]);
        let child = Constraint::from_glrm_grammar(
            "start child; nt child ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                extern g left;
                extern g right;
                nt document ::= "X" left right "!";
            "#,
            &[("left", &child), ("right", &child)],
            &vocab,
        )
        .unwrap();
        let inline = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();

        let mut actual = composed.start();
        let mut expected = inline.start();
        for token_id in [3, 1, 2] {
            assert_eq!(actual.mask(), expected.mask());
            actual.commit_token(token_id).unwrap();
            expected.commit_token(token_id).unwrap();
        }
        assert!(actual.is_complete());
        assert!(expected.is_complete());
    }

    #[test]
    fn glrm_external_subgrammars_support_qualified_nested_bindings() {
        let vocab = vocab(&["<", ">", "[", "]", "a", "<[a]>"]);
        let leaf = Constraint::from_glrm_grammar(
            "start leaf; nt leaf ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                g wrapper ::= {
                    start value;
                    extern g leaf;
                    nt value ::= "[" leaf "]";
                };
                nt document ::= "<" wrapper ">";
            "#,
            &[("wrapper::leaf", &leaf)],
            &vocab,
        )
        .unwrap();

        let mut state = composed.start();
        state.commit_token(5).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn glrm_external_subgrammar_api_preserves_scoped_ignores() {
        let vocab = vocab(&[" ", "\t", "<", ">", "a", "b"]);
        let child = Constraint::from_glrm_grammar(
            r#"
                start value;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt value ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                extern g child;
                nt document ::= "<" child ">";
            "#,
            &[("child", &child)],
            &vocab,
        )
        .unwrap();
        let inline = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start value;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt value ::= "a" "b";
                };
                nt document ::= "<" child ">";
            "#,
            &vocab,
        )
        .unwrap();

        let bytes = b" <\ta\t\tb\t> ";
        let mut actual = composed.start();
        let mut expected = inline.start();
        actual.commit_bytes(bytes).unwrap();
        expected.commit_bytes(bytes).unwrap();
        assert!(actual.is_complete());
        assert!(expected.is_complete());
    }

    #[test]
    fn glrm_external_subgrammar_api_rejects_invalid_bindings() {
        let vocab = vocab(&["a"]);
        let child = Constraint::from_glrm_grammar(
            "start child; nt child ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let source = "start document; extern g child; nt document ::= child;";

        let missing = Constraint::from_glrm_grammar_with_subgrammars(source, &[], &vocab)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("no compiled child"), "{missing}");

        let duplicate = Constraint::from_glrm_grammar_with_subgrammars(
            source,
            &[("child", &child), ("child", &child)],
            &vocab,
        )
        .unwrap_err()
        .to_string();
        assert!(duplicate.contains("more than once"), "{duplicate}");

        let unknown = Constraint::from_glrm_grammar_with_subgrammars(
            source,
            &[("child", &child), ("other", &child)],
            &vocab,
        )
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("unknown external"), "{unknown}");
    }

    #[test]
    fn glrm_external_subgrammar_allocator_avoids_child_special_tokens() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"!".to_vec())]);
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t MARK ::= @token(2);
                nt child ::= "a" MARK;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                extern g child;
                nt document ::= child "!";
            "#,
            &[("child", &child)],
            &vocab,
        )
        .unwrap();

        let mut state = composed.start();
        state.commit_token(0).unwrap();
        state.commit_token(2).unwrap();
        state.commit_token(1).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn external_placeholder_allocator_finds_hole_below_u32_max() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (u32::MAX, b"z".to_vec())]);
        assert_eq!(first_external_placeholder_token_id(&vocab).unwrap(), 1);
    }

    #[test]
    fn typed_glrm_api_without_externals_is_the_normal_compile_path() {
        let vocab = vocab(&["a"]);
        let source = "start document; nt document ::= \"a\";";
        let typed = Constraint::from_glrm_grammar_with_subgrammars(source, &[], &vocab).unwrap();
        let normal = Constraint::from_glrm_grammar(source, &vocab).unwrap();
        assert_eq!(typed.start().mask(), normal.start().mask());
    }

    #[test]
    fn glrm_external_subgrammar_placeholders_avoid_end_tokens() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let child = Constraint::from_glrm_grammar(
            "start child; nt child ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars_and_end_tokens(
            "start document; extern g child; nt document ::= child;",
            &[("child", &child)],
            &vocab,
            &[1],
        )
        .unwrap();
        assert!(
            !composed.table.control_terminals.is_empty(),
            "a child followed directly by an end token must use explicit controls",
        );
        let loaded = Constraint::load(&composed.save()).unwrap();

        for constraint in [&composed, &loaded] {
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            for _ in 0..2 {
                let mask = state.mask();
                assert_ne!(mask[0] & (1 << 1), 0, "end token missing from cached mask");
            }
            state.commit_token(1).unwrap();
            assert!(state.is_complete());
        }
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
