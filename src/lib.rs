#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod grammar;
pub(crate) mod import;
pub(crate) mod runtime;
mod vocab;

pub use ds::weight::{
    clear_all_weights,
    clear_stale_weights,
    clear_weight_caches,
    clear_weight_op_caches,
};
pub use error::{Error, GlrMaskError, Result};
pub use runtime::{
    CommitProfile,
    Constraint,
    ConstraintState,
    FillMaskTimings,
    GssProfileSummary,
    PerAdvanceEntry,
};
pub use vocab::Vocab;

/// Dump a JSON schema as a Lark-like grammar string (for debugging/inspection).
pub fn dump_json_schema_grammar(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let factored = grammar::factoring::factor_named_grammar(named);
    Ok(factored.to_lark())
}

/// Dump a JSON schema grammar in the GLRM format (fully-featured, round-trippable).
///
/// Unreachable rules are pruned before serialisation.
pub fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let factored = grammar::factoring::factor_named_grammar(named);
    let pruned = factored.prune_unreachable();
    Ok(grammar::glrm::to_glrm(&pruned))
}

/// Dump ALL terminals from a JSON schema grammar as JSON.
///
/// Returns a JSON array of objects with `id`, `name` (if available), `type`,
/// and `definition` for each terminal in the lowered grammar.
pub fn dump_json_schema_terminals(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let gdef = grammar::ast::lower(&named)?;

    let mut terminals_json = Vec::new();
    for terminal in &gdef.terminals {
        let id = terminal.id();
        let name = gdef.terminal_names.get(&id).cloned();
        let entry = match terminal {
            grammar::flat::Terminal::Literal { bytes, .. } => {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "type": "literal",
                    "bytes": String::from_utf8_lossy(bytes),
                })
            }
            grammar::flat::Terminal::Pattern { pattern, utf8, .. } => {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "type": "pattern",
                    "pattern": pattern,
                    "utf8": utf8,
                })
            }
            grammar::flat::Terminal::Expr { expr, .. } => {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "type": "expr",
                    "expr_debug": format!("{:?}", expr),
                })
            }
        };
        terminals_json.push(entry);
    }

    let output = serde_json::json!({
        "terminal_count": gdef.terminals.len(),
        "terminals": terminals_json,
    });

    serde_json::to_string_pretty(&output)
        .map_err(|e| GlrMaskError::Serialization(format!("JSON serialization error: {e}")))
}

/// Dump ALL terminals from the JSON schema grammar after the compile-time
/// grammar transforms have run.
///
/// This matches the terminal set used by `compile_owned()` after
/// `prepare_grammar_transforms_only()`.
pub fn dump_json_schema_terminals_prepared(schema_json: &str) -> Result<String> {
    let gdef = import::json_schema::json_schema_to_grammar(schema_json)?;
    let prepared = compiler::grammar::transforms::prepare_grammar_transforms_only(gdef);

    let mut terminals_json = Vec::new();
    for terminal in &prepared.terminals {
        let id = terminal.id();
        let name = prepared.terminal_names.get(&id).cloned();
        let entry = match terminal {
            grammar::flat::Terminal::Literal { bytes, .. } => {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "type": "literal",
                    "bytes": String::from_utf8_lossy(bytes),
                })
            }
            grammar::flat::Terminal::Pattern { pattern, utf8, .. } => {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "type": "pattern",
                    "pattern": pattern,
                    "utf8": utf8,
                })
            }
            grammar::flat::Terminal::Expr { expr, .. } => {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "type": "expr",
                    "expr_debug": format!("{:?}", expr),
                })
            }
        };
        terminals_json.push(entry);
    }

    let output = serde_json::json!({
        "terminal_count": prepared.terminals.len(),
        "terminals": terminals_json,
    });

    serde_json::to_string_pretty(&output)
        .map_err(|e| GlrMaskError::Serialization(format!("JSON serialization error: {e}")))
}

/// Serialize the full GrammarDef for a JSON schema as JSON.
/// This preserves all terminal data (including DFAs) for exact round-tripping.
pub fn dump_json_schema_grammar_def(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let gdef = grammar::ast::lower(&named)?;
    serde_json::to_string(&gdef)
        .map_err(|e| GlrMaskError::Serialization(format!("JSON serialization error: {e}")))
}

/// Serialize the PREPARED GrammarDef (after transforms) for a JSON schema as JSON.
pub fn dump_json_schema_prepared_grammar_def(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let gdef = grammar::ast::lower(&named)?;
    let prepared = compiler::grammar::transforms::prepare_grammar_transforms_only(gdef);
    serde_json::to_string(&prepared)
        .map_err(|e| GlrMaskError::Serialization(format!("JSON serialization error: {e}")))
}

/// Dump the GLR table (action/goto) for a JSON schema as JSON.
pub fn dump_json_schema_glr_table(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let gdef = grammar::ast::lower(&named)?;
    let prepared = compiler::grammar::transforms::prepare_grammar_transforms_only(gdef);
    let analyzed = compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&prepared);
    let table = compiler::glr::table::GLRTable::build(&analyzed);
    serde_json::to_string(&table)
        .map_err(|e| GlrMaskError::Serialization(format!("JSON serialization error: {e}")))
}

/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    Ok(compiler::compile_owned(gdef, vocab))
}
