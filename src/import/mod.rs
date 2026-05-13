pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

use crate::compiler::compile::{compile_owned_profiled, compile_profile_enabled, emit_compile_profile_summary};
use crate::compiler::compile_owned;
use crate::grammar::exact_subtraction_lowering::lower_exact_subtractions;
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::grammar::named_simplify::simplify_named_grammar;
use crate::grammar::terminal_choice_promotion::promote_choice_terminals_exact;
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
        if json_schema::lower_exact_subtractions_enabled() {
            lower_exact_subtractions(&mut factored)?;
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
    if compile_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
    Ok(compile_owned(grammar, vocab))
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
