pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

use std::collections::BTreeSet;

use crate::compiler::compile::{
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_profile_enabled,
    emit_compile_profile_summary,
};
use crate::compiler::pipeline::compile_dynamic_owned_with_table_construction;
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::runtime::Constraint;
use crate::DynamicConstraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;
type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;
type NamedGrammarTransform = fn(&mut ast::NamedGrammar) -> crate::Result<()>;

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
    grammar
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
    if compile_profile_enabled() {
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

fn compile_dynamic_from_source(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<DynamicConstraint> {
    let grammar = lower_factored_named_grammar(source, parse, transform, end_token_ids)?;
    Ok(compile_dynamic_owned_with_table_construction(
        grammar,
        vocab,
        default_table_construction,
    ))
}

/// Profiling-only entry point: runs the JSON-schema import pipeline
/// (parse → factor → AST lower) without the downstream compile. Hidden from the
/// public API; used by `examples/profile_glr.rs` to isolate import timings.
#[doc(hidden)]
pub fn __profile_json_schema_import(schema_json: &str) -> crate::Result<()> {
    let grammar = lower_factored_named_grammar(
        schema_json,
        parse_json_schema_to_named,
        Some(json_schema::prepare_named_grammar),
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
    named
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(
            ebnf,
            vocab,
            "ebnf",
            GlrTableConstruction::ExperimentalCoreMerged,
            ebnf::parse_ebnf_to_named,
            None,
            &[],
        )
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(
            lark,
            vocab,
            "lark",
            GlrTableConstruction::ExperimentalCoreMerged,
            lark::parse_lark_to_named,
            None,
            &[],
        )
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    pub fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        crate::compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
            compile_from_source(
                schema,
                vocab,
                "json_schema",
                GlrTableConstruction::LegacyRowBisim,
                parse_json_schema_to_named,
                Some(json_schema::prepare_named_grammar),
                end_token_ids,
            )
        })
    }

    /// Load a grammar from the GLRM text format.
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(
            glrm,
            vocab,
            "glrm",
            GlrTableConstruction::ExperimentalCoreMerged,
            crate::grammar::glrm::from_glrm,
            None,
            &[],
        )
    }
}

impl DynamicConstraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_dynamic_from_source(
            ebnf,
            vocab,
            GlrTableConstruction::ExperimentalCoreMerged,
            ebnf::parse_ebnf_to_named,
            None,
            &[],
        )
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_dynamic_from_source(
            lark,
            vocab,
            GlrTableConstruction::ExperimentalCoreMerged,
            lark::parse_lark_to_named,
            None,
            &[],
        )
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    pub fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        compile_dynamic_from_source(
            schema,
            vocab,
            GlrTableConstruction::Lalr,
            parse_json_schema_to_named,
            Some(json_schema::prepare_named_grammar),
            end_token_ids,
        )
    }

    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_dynamic_from_source(
            glrm,
            vocab,
            GlrTableConstruction::ExperimentalCoreMerged,
            crate::grammar::glrm::from_glrm,
            None,
            &[],
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::table::{AdmissionPolicy, GlrTableConstruction};
    use crate::Vocab;

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
