//! NOTE: tokenizer regex parsing and construction stay in this split-out file
//! so `Tokenizer` remains focused on runtime stepping.
// SEP1_MAP: sep1 does not keep this as one standalone file; the nearest pieces are spread across `interface/tokenizer_combinators.rs`, `interface/interface.rs`, and regex builders in `finite_automata.rs`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::regex::{Expr, ExprGroup};
use crate::compiler::grammar_def::{GrammarDef, TerminalID};
use crate::ds::u8set::U8Set;

use super::tokenizer::Tokenizer;

impl Tokenizer {
    pub fn from_expr_groups(_groups: &[ExprGroup]) -> Self {
        todo!("tokenizer regex compilation is intentionally deferred during lexer automata cleanup")
    }

    pub fn from_exprs(_terminals: &[(TerminalID, Expr)]) -> Self {
        todo!("tokenizer regex compilation is intentionally deferred during lexer automata cleanup")
    }

    pub fn from_grammar_def(_grammar: &GrammarDef) -> Self {
        todo!("tokenizer regex compilation is intentionally deferred during lexer automata cleanup")
    }
}

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