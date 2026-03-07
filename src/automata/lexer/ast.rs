//! Regular expression AST, compilation (Expr → NFA → DFA), and matching.
//!
//! The [`Expr`] type is the regex AST. It supports:
//! - Literal byte sequences (`U8Seq`)
//! - Character classes (`U8Class`)
//! - Quantifiers (`*`, `+`, `?`)
//! - Bounded repetition (`{n,m}`)
//! - Alternation (`Choice`)
//! - Concatenation (`Seq`)
//! - Epsilon
//!
//! Compilation uses CPS (Continuation-Passing Style) to build NFA fragments,
//! then subset construction + Hopcroft minimization to produce a minimal DFA.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

pub use super::compile::Regex;

/// Quantifier type for regex repetition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QuantifierType {
    /// `*` — zero or more (greedy)
    ZeroOrMore,
    /// `+` — one or more (greedy)
    OneOrMore,
    /// `?` — zero or one
    ZeroOrOne,
}

/// A group in a multi-group regex.
#[derive(Debug, Clone)]
pub struct ExprGroup {
    /// The expression for this group.
    pub expr: Expr,
    /// Whether this group uses non-greedy matching.
    pub is_non_greedy: bool,
}

/// Multiple regex groups compiled together into a single DFA.
///
/// Each group gets a unique group ID. The DFA's finalizer sets indicate
/// which groups match at each accepting state.
#[derive(Debug, Clone)]
pub struct ExprGroups {
    pub groups: Vec<ExprGroup>,
}

/// Regex AST node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    /// A literal byte sequence (e.g., `"abc"`).
    U8Seq(Vec<u8>),
    /// A character class (set of bytes).
    U8Class(U8Set),
    /// Shared sub-expression (for structural sharing / deduplication).
    Shared(Arc<Expr>),
    /// Quantifier: `*`, `+`, or `?` applied to a sub-expression.
    Quantifier(Box<Expr>, QuantifierType),
    /// Bounded repetition: `expr{min,max}`.
    RepeatBounded {
        inner: Box<Expr>,
        min: usize,
        max: Option<usize>,
    },
    /// Alternation: matches any one of the alternatives.
    Choice(Vec<Expr>),
    /// Concatenation: matches the sequence in order.
    Seq(Vec<Expr>),
    /// Epsilon: matches the empty string.
    Epsilon,
}

// ────────────────────────────────── Convenience constructors ──────────────────────────────────

/// Match a single byte.
pub fn byte(b: u8) -> Expr {
    unimplemented!()
}

/// Match a byte sequence.
pub fn bytes(bs: &[u8]) -> Expr {
    unimplemented!()
}

/// Match any byte in a set.
pub fn class(set: U8Set) -> Expr {
    unimplemented!()
}

/// Match any byte in an inclusive range.
pub fn range(lo: u8, hi: u8) -> Expr {
    unimplemented!()
}

/// Zero or more (Kleene star).
pub fn star(e: impl Into<Expr>) -> Expr {
    unimplemented!()
}

/// One or more.
pub fn plus(e: impl Into<Expr>) -> Expr {
    unimplemented!()
}

/// Zero or one.
pub fn opt(e: impl Into<Expr>) -> Expr {
    unimplemented!()
}

/// Bounded repetition `expr{min,max}`.
pub fn repeat(e: impl Into<Expr>, min: usize, max: Option<usize>) -> Expr {
    unimplemented!()
}

/// Alternation.
pub fn choice(alts: Vec<Expr>) -> Expr {
    unimplemented!()
}

/// Concatenation.
pub fn seq(parts: Vec<Expr>) -> Expr {
    unimplemented!()
}

/// Epsilon (empty string).
pub fn eps() -> Expr {
    unimplemented!()
}

impl From<u8> for Expr {
    fn from(b: u8) -> Self {
        unimplemented!()
    }
}

impl From<&[u8]> for Expr {
    fn from(bs: &[u8]) -> Self {
        unimplemented!()
    }
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        unimplemented!()
    }
}

// ────────────────────────────────── Tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        let r = bytes(b"hello").build();
        assert!(r.is_match(b"hello"));
        assert!(!r.is_match(b"hell"));
        assert!(!r.is_match(b"helloo"));
        assert!(!r.is_match(b""));
    }

    #[test]
    fn test_quantifier_star() {
        // a*
        let r = star(byte(b'a')).build();
        assert!(r.is_match(b""));
        assert!(r.is_match(b"a"));
        assert!(r.is_match(b"aaa"));
        assert!(!r.is_match(b"b"));
        assert!(!r.is_match(b"ab"));
    }

    #[test]
    fn test_quantifier_plus() {
        // a+
        let r = plus(byte(b'a')).build();
        assert!(!r.is_match(b""));
        assert!(r.is_match(b"a"));
        assert!(r.is_match(b"aaa"));
        assert!(!r.is_match(b"b"));
    }

    #[test]
    fn test_quantifier_opt() {
        // a?
        let r = opt(byte(b'a')).build();
        assert!(r.is_match(b""));
        assert!(r.is_match(b"a"));
        assert!(!r.is_match(b"aa"));
    }

    #[test]
    fn test_choice() {
        // a|b|c
        let r = choice(vec![byte(b'a'), byte(b'b'), byte(b'c')]).build();
        assert!(r.is_match(b"a"));
        assert!(r.is_match(b"b"));
        assert!(r.is_match(b"c"));
        assert!(!r.is_match(b"d"));
        assert!(!r.is_match(b""));
        assert!(!r.is_match(b"ab"));
    }

    #[test]
    fn test_seq() {
        // abc
        let r = seq(vec![byte(b'a'), byte(b'b'), byte(b'c')]).build();
        assert!(r.is_match(b"abc"));
        assert!(!r.is_match(b"ab"));
        assert!(!r.is_match(b"abcd"));
    }

    #[test]
    fn test_class() {
        // [a-z]
        let r = class(U8Set::from_range(b'a', b'z')).build();
        assert!(r.is_match(b"a"));
        assert!(r.is_match(b"z"));
        assert!(!r.is_match(b"A"));
        assert!(!r.is_match(b"0"));
        assert!(!r.is_match(b""));
        assert!(!r.is_match(b"ab"));
    }

    #[test]
    fn test_repeat_bounded() {
        // a{2,4}
        let r = repeat(byte(b'a'), 2, Some(4)).build();
        assert!(!r.is_match(b"a"));
        assert!(r.is_match(b"aa"));
        assert!(r.is_match(b"aaa"));
        assert!(r.is_match(b"aaaa"));
        assert!(!r.is_match(b"aaaaa"));
        assert!(!r.is_match(b""));
    }

    #[test]
    fn test_repeat_unbounded() {
        // a{2,}
        let r = repeat(byte(b'a'), 2, None).build();
        assert!(!r.is_match(b"a"));
        assert!(r.is_match(b"aa"));
        assert!(r.is_match(b"aaa"));
        assert!(r.is_match(b"aaaaaaaaaa"));
        assert!(!r.is_match(b""));
    }

    #[test]
    fn test_repeat_exact() {
        // a{3,3}
        let r = repeat(byte(b'a'), 3, Some(3)).build();
        assert!(r.is_match(b"aaa"));
        assert!(!r.is_match(b"aa"));
        assert!(!r.is_match(b"aaaa"));
    }

    #[test]
    fn test_complex_pattern() {
        // [a-z]+[0-9]*
        let r = seq(vec![
            plus(class(U8Set::from_range(b'a', b'z'))),
            star(class(U8Set::from_range(b'0', b'9'))),
        ])
        .build();
        assert!(r.is_match(b"hello"));
        assert!(r.is_match(b"hello123"));
        assert!(r.is_match(b"a"));
        assert!(!r.is_match(b"123"));
        assert!(!r.is_match(b""));
    }

    #[test]
    fn test_nested_quantifiers() {
        // (a*b)*
        let r = star(seq(vec![star(byte(b'a')), byte(b'b')])).build();
        assert!(r.is_match(b""));
        assert!(r.is_match(b"b"));
        assert!(r.is_match(b"ab"));
        assert!(r.is_match(b"aab"));
        assert!(r.is_match(b"bab"));
        assert!(r.is_match(b"aabab"));
        assert!(!r.is_match(b"a"));
    }

    #[test]
    fn test_epsilon() {
        let r = eps().build();
        assert!(r.is_match(b""));
        assert!(!r.is_match(b"a"));
    }

    #[test]
    fn test_multi_group() {
        let regex = ExprGroups {
            groups: vec![
                ExprGroup {
                    expr: bytes(b"abc"),
                    is_non_greedy: false,
                },
                ExprGroup {
                    expr: bytes(b"def"),
                    is_non_greedy: false,
                },
                ExprGroup {
                    expr: bytes(b"ab"),
                    is_non_greedy: false,
                },
            ],
        }
        .build();

        let m1 = regex.dfa.find_matches(b"abc");
        assert!(m1.contains(&0));
        assert!(!m1.contains(&1));

        let m2 = regex.dfa.find_matches(b"def");
        assert!(m2.contains(&1));

        let m3 = regex.dfa.find_matches(b"ab");
        assert!(m3.contains(&2));
        assert!(!m3.contains(&0)); // "abc" needs 3 chars
    }

    #[test]
    fn test_non_greedy_metadata_survives_compilation() {
        let regex = ExprGroups {
            groups: vec![
                ExprGroup {
                    expr: bytes(b"a"),
                    is_non_greedy: true,
                },
                ExprGroup {
                    expr: bytes(b"ab"),
                    is_non_greedy: false,
                },
            ],
        }
        .build();

        let state_after_a = regex.dfa.run(b"a");
        assert!(regex.dfa.finalizers(state_after_a).contains(&0));
        assert!(regex.dfa.non_greedy_finalizers(state_after_a).contains(&0));
        assert!(regex
            .dfa
            .possible_future_group_ids(state_after_a)
            .contains(&1));
    }

    #[test]
    fn test_choice_with_empty() {
        // a | epsilon
        let r = choice(vec![byte(b'a'), eps()]).build();
        assert!(r.is_match(b""));
        assert!(r.is_match(b"a"));
        assert!(!r.is_match(b"aa"));
    }

    #[test]
    fn test_overlapping_classes() {
        // [a-m] | [g-z]
        let r = choice(vec![
            class(U8Set::from_range(b'a', b'm')),
            class(U8Set::from_range(b'g', b'z')),
        ])
        .build();
        assert!(r.is_match(b"a"));
        assert!(r.is_match(b"g")); // overlapping
        assert!(r.is_match(b"z"));
        assert!(!r.is_match(b"0"));
    }

    #[test]
    fn test_shared_subexpression() {
        let shared_part = Arc::new(seq(vec![byte(b'a'), byte(b'b')]));

        let r = choice(vec![
            seq(vec![Expr::Shared(shared_part.clone()), byte(b'c')]),
            seq(vec![Expr::Shared(shared_part.clone()), byte(b'd')]),
        ])
        .build();

        assert!(r.is_match(b"abc"));
        assert!(r.is_match(b"abd"));
        assert!(!r.is_match(b"ab"));
        assert!(!r.is_match(b"abe"));
    }

    #[test]
    fn test_regex_step() {
        let r = bytes(b"ab").build();
        let s1 = r.step(0, b'a');
        assert!(s1.is_some());
        let s2 = r.step(s1.unwrap(), b'b');
        assert!(s2.is_some());
        assert!(r.is_accepting(s2.unwrap()));
    }

    #[test]
    fn test_lots_of_words() {
        let words: Vec<&[u8]> = vec![
            b"if",
            b"else",
            b"while",
            b"for",
            b"return",
            b"break",
            b"continue",
            b"int",
            b"float",
            b"void",
            b"char",
        ];
        let groups: Vec<ExprGroup> = words
            .iter()
            .map(|w| ExprGroup {
                expr: bytes(w),
                is_non_greedy: false,
            })
            .collect();
        let regex = ExprGroups { groups }.build();

        for (i, word) in words.iter().enumerate() {
            let matches = regex.dfa.find_matches(word);
            assert!(
                matches.contains(&i),
                "word {:?} should match group {}",
                std::str::from_utf8(word).unwrap(),
                i
            );
        }

        // Non-words should not match
        assert!(regex.dfa.find_matches(b"foo").is_empty());
        assert!(regex.dfa.find_matches(b"").is_empty());
    }
}
