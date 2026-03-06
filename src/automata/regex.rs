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

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

use super::dfa::Dfa;
use super::nfa::Nfa;

/// A compiled regex (wraps a minimized DFA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    /// The underlying DFA.
    pub dfa: Dfa,
}

impl Regex {
    /// Number of states in the DFA.
    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    /// Whether the regex matches the input completely.
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.dfa.accepts(input)
    }

    /// Get the next DFA state for a byte, starting from the given state.
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    /// Whether a DFA state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.dfa.is_accepting(state)
    }

    /// Get the set of valid next bytes from a DFA state.
    pub fn get_u8set(&self, state: u32) -> U8Set {
        self.dfa.get_u8set(state)
    }
}

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
    Expr::U8Seq(vec![b])
}

/// Match a byte sequence.
pub fn bytes(bs: &[u8]) -> Expr {
    Expr::U8Seq(bs.to_vec())
}

/// Match any byte in a set.
pub fn class(set: U8Set) -> Expr {
    Expr::U8Class(set)
}

/// Match any byte in an inclusive range.
pub fn range(lo: u8, hi: u8) -> Expr {
    Expr::U8Class(U8Set::from_range(lo, hi))
}

/// Zero or more (Kleene star).
pub fn star(e: impl Into<Expr>) -> Expr {
    Expr::Quantifier(Box::new(e.into()), QuantifierType::ZeroOrMore)
}

/// One or more.
pub fn plus(e: impl Into<Expr>) -> Expr {
    Expr::Quantifier(Box::new(e.into()), QuantifierType::OneOrMore)
}

/// Zero or one.
pub fn opt(e: impl Into<Expr>) -> Expr {
    Expr::Quantifier(Box::new(e.into()), QuantifierType::ZeroOrOne)
}

/// Bounded repetition `expr{min,max}`.
pub fn repeat(e: impl Into<Expr>, min: usize, max: Option<usize>) -> Expr {
    Expr::RepeatBounded {
        inner: Box::new(e.into()),
        min,
        max,
    }
}

/// Alternation.
pub fn choice(alts: Vec<Expr>) -> Expr {
    Expr::Choice(alts)
}

/// Concatenation.
pub fn seq(parts: Vec<Expr>) -> Expr {
    Expr::Seq(parts)
}

/// Epsilon (empty string).
pub fn eps() -> Expr {
    Expr::Epsilon
}

impl From<u8> for Expr {
    fn from(b: u8) -> Self {
        Expr::U8Seq(vec![b])
    }
}

impl From<&[u8]> for Expr {
    fn from(bs: &[u8]) -> Self {
        Expr::U8Seq(bs.to_vec())
    }
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        Expr::U8Seq(s.as_bytes().to_vec())
    }
}

// ────────────────────────────────── Compilation ──────────────────────────────────

impl Expr {
    /// Build a single-group regex from this expression.
    pub fn build(self) -> Regex {
        ExprGroups {
            groups: vec![ExprGroup {
                expr: self,
                is_non_greedy: false,
            }],
        }
        .build()
    }

    /// CPS (Continuation-Passing Style) NFA compilation.
    ///
    /// Compiles this expression into NFA states such that entering at the returned
    /// state will recognize the expression and then flow to `cont`.
    fn compile_cps(
        expr: &Expr,
        nfa: &mut Nfa,
        cont: u32,
        cache: &mut HashMap<(usize, u32), u32>,
    ) -> u32 {
        match expr {
            Expr::Epsilon => cont,

            Expr::U8Seq(bs) => {
                if bs.is_empty() {
                    return cont;
                }
                // Build chain backwards: last byte → cont, then second-to-last → last, etc.
                let mut s = cont;
                for &b in bs.iter().rev() {
                    let p = nfa.add_state();
                    nfa.add_transition(p, b, s);
                    s = p;
                }
                s
            }

            Expr::U8Class(set) => {
                let s = nfa.add_state();
                nfa.add_u8set_transition(s, *set, cont);
                s
            }

            Expr::Seq(children) => {
                if children.is_empty() {
                    return cont;
                }
                let mut s = cont;
                for child in children.iter().rev() {
                    s = Self::compile_cps(child, nfa, s, cache);
                }
                s
            }

            Expr::Choice(alts) => {
                if alts.is_empty() {
                    // Empty choice matches nothing — dead state
                    return nfa.add_state();
                }
                if alts.len() == 1 {
                    return Self::compile_cps(&alts[0], nfa, cont, cache);
                }
                // Create a split state with epsilon transitions to each alternative
                let split = nfa.add_state();
                for alt in alts {
                    let alt_start = Self::compile_cps(alt, nfa, cont, cache);
                    nfa.add_epsilon(split, alt_start);
                }
                split
            }

            Expr::Quantifier(inner, qtype) => match qtype {
                QuantifierType::ZeroOrMore => {
                    // a* : split → (inner → loop_back) | cont
                    let split = nfa.add_state();
                    let body_start = Self::compile_cps(inner, nfa, split, cache);
                    nfa.add_epsilon(split, body_start); // match one more
                    nfa.add_epsilon(split, cont); // or skip
                    split
                }
                QuantifierType::OneOrMore => {
                    // a+ : body → split → (body | cont)
                    let split = nfa.add_state();
                    let body_start = Self::compile_cps(inner, nfa, split, cache);
                    nfa.add_epsilon(split, body_start); // match one more
                    nfa.add_epsilon(split, cont); // or done
                    body_start // must match at least once
                }
                QuantifierType::ZeroOrOne => {
                    // a? : split → (inner → cont) | cont
                    let split = nfa.add_state();
                    let body_start = Self::compile_cps(inner, nfa, cont, cache);
                    nfa.add_epsilon(split, body_start);
                    nfa.add_epsilon(split, cont);
                    split
                }
            },

            Expr::RepeatBounded { inner, min, max } => {
                // expr{min,max}
                // Build: min mandatory copies, then (max-min) optional copies
                let mut s = cont;

                // Optional copies (from max down to min)
                if let Some(max_val) = max {
                    let optional = max_val.saturating_sub(*min);
                    for _ in 0..optional {
                        // Each optional copy: split → (inner → next) | next
                        let split = nfa.add_state();
                        let body_start = Self::compile_cps(inner, nfa, s, cache);
                        nfa.add_epsilon(split, body_start);
                        nfa.add_epsilon(split, s); // skip
                        s = split;
                    }
                } else {
                    // Unbounded: after min copies, do inner*
                    // star_split → (inner → star_split) | next
                    let star_split = nfa.add_state();
                    let body_start = Self::compile_cps(inner, nfa, star_split, cache);
                    nfa.add_epsilon(star_split, body_start);
                    nfa.add_epsilon(star_split, s);
                    s = star_split;
                }

                // Mandatory copies
                for _ in 0..*min {
                    s = Self::compile_cps(inner, nfa, s, cache);
                }

                s
            }

            Expr::Shared(arc) => {
                let key = (Arc::as_ptr(arc) as usize, cont);
                if let Some(&cached) = cache.get(&key) {
                    return cached;
                }
                let result = Self::compile_cps(arc.as_ref(), nfa, cont, cache);
                cache.insert(key, result);
                result
            }
        }
    }
}

impl ExprGroups {
    /// Compile all groups into a single multi-group regex.
    pub fn build(self) -> Regex {
        let nfa = self.build_nfa();
        let dfa = nfa.to_dfa();
        let dfa = dfa.minimize();
        Regex { dfa }
    }

    /// Compile to NFA (without DFA conversion — useful for testing).
    pub fn build_nfa(self) -> Nfa {
        // Start with a split state (state 0) that branches to each group
        let mut nfa = Nfa::new(1); // state 0 = split point
        let mut cache = HashMap::new();

        for (
            group_idx,
            ExprGroup {
                expr,
                is_non_greedy,
            },
        ) in self.groups.into_iter().enumerate()
        {
            // Create accept state for this group
            let accept = nfa.add_state();
            nfa.add_finalizer(accept, group_idx);
            if is_non_greedy {
                nfa.add_non_greedy_finalizer(accept, group_idx);
            }

            // Compile the expression into the accept state
            let group_start = Expr::compile_cps(&expr, &mut nfa, accept, &mut cache);

            // Connect split state to this group's start
            nfa.add_epsilon(0, group_start);
        }

        nfa
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
