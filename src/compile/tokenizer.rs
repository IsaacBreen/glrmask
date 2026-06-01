//! Tokenizer construction from the normalized grammar.
//!
//! The tokenizer is the lexer DFA used by both compile-time automata and the
//! runtime scanner.  It is built from the grammar terminals, not from the LLM
//! vocabulary.  This is a key boundary: vocabulary-dependent quotienting lives
//! in Terminal-DWA and scan-relation phases, while the tokenizer itself is a
//! grammar object.

use crate::Vocab;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::compile::options::tokenizer_detail_profile_enabled;
use crate::compile::profiling::elapsed_ms;
use crate::grammar::flat::{GrammarDef, Terminal};
use std::time::Instant;

/// Build the grammar tokenizer and isolate runtime boundary states.
pub(crate) fn build_tokenizer(grammar: &GrammarDef) -> Tokenizer {
    let exprs: Vec<Expr> = grammar.terminals.iter().map(terminal_expr).collect();
    if tokenizer_detail_profile_enabled() {
        emit_tokenizer_detail(grammar, &exprs);
    }
    build_tokenizer_from_exprs(&exprs)
}

/// Build a tokenizer directly from terminal expressions.
pub(crate) fn build_tokenizer_from_exprs(exprs: &[Expr]) -> Tokenizer {
    let regex = build_regex(exprs);

    Tokenizer {
        dfa: regex.dfa,
        num_terminals: exprs.len() as u32,
        exprs: Some(std::sync::Arc::from(exprs.to_vec())),
    }
}

fn terminal_expr(terminal: &Terminal) -> Expr {
    match terminal {
        Terminal::Literal { bytes, .. } => Expr::U8Seq(bytes.clone()),
        Terminal::Pattern { pattern, utf8, .. } => parse_regex(pattern, *utf8),
        Terminal::Expr { expr, .. } => expr.clone(),
    }
}

fn emit_tokenizer_detail(grammar: &GrammarDef, exprs: &[Expr]) {
    eprintln!(
        "[glrmask/profile][tokenizer] terminals={}",
        grammar.terminals.len()
    );
    for (index, expr) in exprs.iter().enumerate() {
        let started_at = Instant::now();
        let regex = build_regex(std::slice::from_ref(expr));
        let elapsed = elapsed_ms(started_at);
        let name = grammar.terminal_display_name(index as u32);
        eprintln!(
            "[glrmask/profile][tokenizer] terminal id={} name={:?} final_states={} final_transitions={} alone_ms={:.3}",
            index,
            name,
            regex.num_states(),
            regex.num_transitions(),
            elapsed
        );
    }
}

/// Documentation hook: the tokenizer phase depends on the vocabulary only after
/// construction, when downstream phases classify token bytes against it.  This
/// function exists so future compile options can make that dependency explicit.
pub(crate) fn tokenizer_is_independent_of_vocab(_: &GrammarDef, _: &Vocab) -> bool {
    true
}
