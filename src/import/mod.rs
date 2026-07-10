pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

use crate::compiler::compile::{
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_profile_enabled,
    emit_compile_profile_summary,
};
use crate::grammar::exact_subtraction_lowering::lower_exact_subtractions;
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::grammar::named_simplify::simplify_named_grammar;
use crate::grammar::terminal_choice_promotion::promote_choice_terminals_exact;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::runtime::Constraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;
type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;

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
    source_kind: &str,
    parse_named: NamedGrammarParser,
) -> crate::Result<GrammarDef> {
    let lower_started_at = emit_import_phase_start("lower_factored_named_grammar");
    let parse_named_started_at = emit_import_phase_start("parse_named");
    let named = parse_named(source)?;
    emit_import_phase_end("parse_named", parse_named_started_at);

    let factor_started_at = emit_import_phase_start("factor_named_grammar");
    let mut factored = factor_named_grammar(named);
    emit_import_phase_end("factor_named_grammar", factor_started_at);

    if source_kind == "json_schema" {
        if json_schema::simplify_grammar_enabled() {
            let simplify_started_at = emit_import_phase_start("simplify_named_grammar");
            simplify_named_grammar(&mut factored);
            emit_import_phase_end("simplify_named_grammar", simplify_started_at);
        }
        if json_schema::lower_exact_subtractions_enabled() {
            let lower_exact_started_at = emit_import_phase_start("lower_exact_subtractions");
            lower_exact_subtractions(&mut factored)?;
            emit_import_phase_end("lower_exact_subtractions", lower_exact_started_at);
        }
        if json_schema::promote_literal_choices_enabled() {
            let promote_started_at = emit_import_phase_start("promote_choice_terminals_exact");
            promote_choice_terminals_exact(&mut factored, false);
            emit_import_phase_end("promote_choice_terminals_exact", promote_started_at);
        }
        json_schema::assign_default_lexer_partitions(&mut factored);
    }

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
) -> crate::Result<Constraint> {
    let compile_from_source_started_at = emit_import_phase_start("compile_from_source");
    if compile_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = compile_owned_profiled_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        );
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        emit_import_phase_end("compile_from_source", compile_from_source_started_at);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
    let constraint = compile_owned_with_table_construction(
        grammar,
        vocab,
        default_table_construction,
    );
    emit_import_phase_end("compile_from_source", compile_from_source_started_at);
    Ok(constraint)
}

/// Profiling-only entry point: runs the JSON-schema import pipeline
/// (parse → factor → AST lower) without the downstream compile. Hidden from the
/// public API; used by `examples/profile_glr.rs` to isolate import timings.
#[doc(hidden)]
pub fn __profile_json_schema_import(schema_json: &str) -> crate::Result<()> {
    let grammar = lower_factored_named_grammar(
        schema_json,
        "json_schema",
        parse_json_schema_to_named,
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
        )
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(
            lark,
            vocab,
            "lark",
            GlrTableConstruction::ExperimentalCoreMerged,
            lark::parse_lark_to_named,
        )
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(
            schema,
            vocab,
            "json_schema",
            GlrTableConstruction::LegacyRowBisim,
            parse_json_schema_to_named,
        )
    }

    /// Load a grammar from the GLRM format (see [`crate::grammar::glrm`]).
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(
            glrm,
            vocab,
            "glrm",
            GlrTableConstruction::ExperimentalCoreMerged,
            crate::grammar::glrm::from_glrm,
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
                .collect(),
            None,
        )
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
