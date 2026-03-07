//! Tokenizer regex parsing / construction helpers.
//!
//! This file owns the regex-facing builder surface so `Tokenizer` itself can
//! stay focused on runtime stepping/execution behavior.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::regex::{Expr, ExprGroup};
use crate::compiler::grammar_def::{GrammarDef, TerminalID};
use crate::ds::u8set::U8Set;

use super::tokenizer::Tokenizer;

impl Tokenizer {
    /// Build a tokenizer DFA from fully specified regex groups.
    pub fn from_expr_groups(_groups: &[ExprGroup]) -> Self {
        todo!("tokenizer regex compilation is intentionally deferred during lexer automata cleanup")
    }

    /// Build a tokenizer DFA from terminal expressions.
    pub fn from_exprs(_terminals: &[(TerminalID, Expr)]) -> Self {
        todo!("tokenizer regex compilation is intentionally deferred during lexer automata cleanup")
    }

    /// Build a tokenizer DFA from a `GrammarDef` by parsing terminal patterns.
    pub fn from_grammar_def(_grammar: &GrammarDef) -> Self {
        todo!("tokenizer regex compilation is intentionally deferred during lexer automata cleanup")
    }
}

/// Parse a simple regex pattern string into an `Expr` AST.
pub fn parse_regex(_pattern: &str) -> Expr {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_alternation(_input: &[u8], _pos: usize) -> (Expr, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_sequence(_input: &[u8], _pos: usize) -> (Expr, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_quantified(_input: &[u8], _pos: usize) -> (Expr, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_repetition_bounds(_input: &[u8], _pos: usize) -> (usize, Option<usize>, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_usize(_input: &[u8], _pos: usize) -> (usize, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_atom(_input: &[u8], _pos: usize) -> (Expr, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_char_class(_input: &[u8], _pos: usize) -> (Expr, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_escape(_input: &[u8], _pos: usize) -> (Expr, usize) {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn parse_escape_byte(_input: &[u8], _pos: usize) -> u8 {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn escape_len(_input: &[u8], _pos: usize) -> usize {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}

fn hex_digit(_b: u8) -> u8 {
    todo!("tokenizer regex parsing is intentionally deferred during lexer automata cleanup")
}