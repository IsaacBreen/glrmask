//! Regular-expression AST surface for the lexer pipeline.
//!
//! This file now keeps only the structural regex representation and the small
//! helper constructors that other lexer-side code still refers to. The old
//! Regex/DFA-behavior tests that exercised removed helper surface are
//! intentionally omitted until the sep1-style lexer rewrite lands.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    U8Seq(Vec<u8>),
    U8Class(U8Set),
    Seq(Vec<Expr>),
    Choice(Vec<Expr>),
    Repeat {
        expr: Box<Expr>,
        min: usize,
        max: Option<usize>,
    },
    Shared(Arc<Expr>),
    Epsilon,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprGroup {
    pub expr: Expr,
    pub is_non_greedy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExprGroups {
    pub groups: Vec<ExprGroup>,
}

pub fn byte(b: u8) -> Expr {
    Expr::U8Seq(vec![b])
}

pub fn bytes(bs: &[u8]) -> Expr {
    Expr::U8Seq(bs.to_vec())
}

pub fn class(set: U8Set) -> Expr {
    Expr::U8Class(set)
}

pub fn seq(exprs: Vec<Expr>) -> Expr {
    Expr::Seq(exprs)
}

pub fn choice(exprs: Vec<Expr>) -> Expr {
    Expr::Choice(exprs)
}

pub fn repeat(expr: impl Into<Expr>, min: usize, max: Option<usize>) -> Expr {
    Expr::Repeat {
        expr: Box::new(expr.into()),
        min,
        max,
    }
}

pub fn star(expr: impl Into<Expr>) -> Expr {
    repeat(expr, 0, None)
}

pub fn plus(expr: impl Into<Expr>) -> Expr {
    repeat(expr, 1, None)
}

pub fn opt(expr: impl Into<Expr>) -> Expr {
    repeat(expr, 0, Some(1))
}

pub fn eps() -> Expr {
    Expr::Epsilon
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        Expr::U8Seq(s.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    // The old AST tests were tightly coupled to removed Regex / DFA helper
    // surface and are intentionally omitted until the sep1-style lexer rewrite
    // lands.
}
