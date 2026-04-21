//! Regex AST and constructor surface.
//!
//! These constructors form the public "builder" API for regex expressions.
//! Even if some (like `seq`, `star`, `opt`) have no in-tree callers right now,
//! they are retained deliberately: every `Expr` variant should have a
//! corresponding named constructor so the API surface stays coherent and
//! usable for tests, downstream crate consumers, and future grammar work.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expr {
    U8Seq(Vec<u8>),
    U8Class(U8Set),
    Intersect {
        expr: Box<Expr>,
        intersect: Box<Expr>,
    },
    Seq(Vec<Expr>),
    Choice(Vec<Expr>),
    Exclude {
        expr: Box<Expr>,
        exclude: Box<Expr>,
    },
    Repeat {
        expr: Box<Expr>,
        min: usize,
        max: Option<usize>,
    },
    Shared(Arc<Expr>),
    Epsilon,
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

pub fn exclude(expr: impl Into<Expr>, excluded: impl Into<Expr>) -> Expr {
    Expr::Exclude {
        expr: Box::new(expr.into()),
        exclude: Box::new(excluded.into()),
    }
}

pub fn intersect(expr: impl Into<Expr>, other: impl Into<Expr>) -> Expr {
    Expr::Intersect {
        expr: Box::new(expr.into()),
        intersect: Box::new(other.into()),
    }
}

pub fn repeat(expr: impl Into<Expr>, min: usize, max: Option<usize>) -> Expr {
    Expr::Repeat {
        expr: Box::new(expr.into()),
        min,
        max,
    }
}

pub fn plus(expr: impl Into<Expr>) -> Expr {
    repeat(expr, 1, None)
}

pub fn star(expr: impl Into<Expr>) -> Expr {
    repeat(expr, 0, None)
}

pub fn opt(expr: impl Into<Expr>) -> Expr {
    repeat(expr, 0, Some(1))
}

pub fn eps() -> Expr {
    Expr::Epsilon
}

fn optimize_repeat_expr(expr: Expr, min: usize, max: Option<usize>) -> Expr {
    match (min, max) {
        (0, Some(0)) => Expr::Epsilon,
        (1, Some(1)) => expr,
        _ => Expr::Repeat {
            expr: Box::new(expr),
            min,
            max,
        },
    }
}

fn flatten_sequence_parts(exprs: Vec<Expr>) -> Vec<Expr> {
    let mut flat = Vec::with_capacity(exprs.len());
    for expr in exprs {
        match expr {
            Expr::Seq(children) => flat.extend(children),
            Expr::Epsilon => {}
            other => flat.push(other),
        }
    }
    flat
}

fn simplify_single_byte_classes(exprs: &mut [Expr]) {
    for expr in exprs {
        if let Expr::U8Class(set) = expr {
            if set.len() == 1 {
                let byte = set.iter().next().unwrap();
                *expr = Expr::U8Seq(vec![byte]);
            }
        }
    }
}

fn merge_adjacent_byte_sequences(exprs: Vec<Expr>) -> Vec<Expr> {
    let mut merged = Vec::with_capacity(exprs.len());
    for expr in exprs {
        match expr {
            Expr::U8Seq(mut current) => {
                if let Some(Expr::U8Seq(previous)) = merged.last_mut() {
                    previous.append(&mut current);
                } else {
                    merged.push(Expr::U8Seq(current));
                }
            }
            other => merged.push(other),
        }
    }
    merged
}

fn flatten_choice_parts(exprs: Vec<Expr>) -> Vec<Expr> {
    let mut worklist = exprs;
    let mut flat = Vec::with_capacity(worklist.len());
    while let Some(expr) = worklist.pop() {
        match expr {
            Expr::Choice(children) => worklist.extend(children),
            other => flat.push(other),
        }
    }
    flat
}

fn fold_choice_byte_classes(exprs: Vec<Expr>) -> Vec<Expr> {
    let mut classes = U8Set::empty();
    let mut complex = Vec::with_capacity(exprs.len());
    for expr in exprs {
        match expr {
            Expr::U8Class(set) => classes |= set,
            Expr::U8Seq(bytes) if bytes.len() == 1 => {
                classes.insert(bytes[0]);
            }
            other => complex.push(other),
        }
    }

    if !classes.is_empty() {
        complex.push(Expr::U8Class(classes));
    }

    complex
}

impl Expr {
    pub fn is_nullable(&self) -> bool {
        match self {
            Expr::U8Seq(bytes) => bytes.is_empty(),
            Expr::U8Class(_) => false,
            Expr::Intersect { expr, intersect } => expr.is_nullable() && intersect.is_nullable(),
            Expr::Seq(parts) => parts.iter().all(Expr::is_nullable),
            Expr::Choice(options) => options.iter().any(Expr::is_nullable),
            Expr::Exclude { expr, exclude } => expr.is_nullable() && !exclude.is_nullable(),
            Expr::Repeat { expr, min, .. } => *min == 0 || expr.is_nullable(),
            Expr::Shared(expr) => expr.is_nullable(),
            Expr::Epsilon => true,
        }
    }

    pub fn optimize(self) -> Self {
        match self {
            Expr::Seq(parts) => Expr::make_seq(parts.into_iter().map(Expr::optimize).collect()),
            Expr::Choice(options) => {
                Expr::make_choice(options.into_iter().map(Expr::optimize).collect())
            }
            Expr::Intersect { expr, intersect } => Expr::Intersect {
                expr: Box::new(expr.optimize()),
                intersect: Box::new(intersect.optimize()),
            },
            Expr::Exclude { expr, exclude } => Expr::Exclude {
                expr: Box::new(expr.optimize()),
                exclude: Box::new(exclude.optimize()),
            },
            Expr::Repeat { expr, min, max } => {
                let child = expr.optimize();
                optimize_repeat_expr(child, min, max)
            }
            Expr::Shared(inner) => Expr::Shared(Arc::new(inner.as_ref().clone().optimize())),
            leaf => leaf,
        }
    }

    pub fn strip_prefix(&self, prefix: &Expr) -> Option<Expr> {
        if self == prefix {
            return Some(Expr::Epsilon);
        }

        match self {
            Expr::Seq(exprs) => {
                if exprs.is_empty() {
                    return None;
                }
                if &exprs[0] == prefix {
                    return Some(Expr::make_seq(exprs[1..].to_vec()));
                }
                None
            }
            Expr::Intersect { .. } => None,
            Expr::Exclude { .. } => None,
            Expr::Shared(inner) => inner.strip_prefix(prefix),
            _ => None,
        }
    }

    pub fn make_seq(exprs: Vec<Expr>) -> Expr {
        let mut flat = flatten_sequence_parts(exprs);

        if flat.is_empty() {
            return Expr::Epsilon;
        }

        simplify_single_byte_classes(&mut flat);
        let mut merged = merge_adjacent_byte_sequences(flat);

        if merged.len() == 1 {
            merged.pop().unwrap()
        } else {
            Expr::Seq(merged)
        }
    }

    pub fn make_choice(exprs: Vec<Expr>) -> Expr {
        let mut flat = flatten_choice_parts(exprs);

        if flat.is_empty() {
            return Expr::Choice(vec![]);
        }
        if flat.len() == 1 {
            return flat.pop().unwrap();
        }

        let mut complex = fold_choice_byte_classes(flat);

        if complex.len() == 1 {
            complex.pop().unwrap()
        } else {
            Expr::Choice(complex)
        }
    }
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        Expr::U8Seq(s.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    // NOTE: the old AST tests are intentionally omitted until the legacy
    // lexer rewrite lands.
}
