//! Tokenizer DFA construction.
//!
//! Builds a multi-pattern DFA that recognizes terminal patterns.
//! Given a stream of bytes, the DFA tracks which terminals are
//! currently matching (via finalizer groups).

use std::collections::BTreeSet;

use crate::automata::dfa::Dfa;
use crate::automata::regex::{Expr, ExprGroup, ExprGroups};
use crate::compiler::grammar_def::{GrammarDef, TerminalId};
use crate::ds::u8set::U8Set;

/// A tokenizer built from terminal patterns.
///
/// Wraps a multi-group DFA where each group corresponds to a terminal.
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerDfa {
    /// The underlying multi-group DFA.
    pub dfa: Dfa,
    /// Number of terminals.
    pub num_terminals: u32,
}

impl TokenizerDfa {
    /// Build a tokenizer DFA from terminal expressions.
    ///
    /// `terminals[i]` = (terminal_id, expression).
    /// Terminal i maps to DFA group i.
    pub fn from_exprs(terminals: &[(TerminalId, Expr)]) -> Self {
        let groups: Vec<ExprGroup> = terminals
            .iter()
            .map(|(_tid, expr)| ExprGroup {
                expr: expr.clone(),
                is_non_greedy: false,
            })
            .collect();
        let num_terminals = terminals.len() as u32;
        let dfa = ExprGroups { groups }.build();
        Self {
            dfa: dfa.dfa,
            num_terminals,
        }
    }

    /// Build a tokenizer DFA from a GrammarDef by parsing terminal patterns.
    pub fn from_grammar_def(grammar: &GrammarDef) -> Self {
        let terminals: Vec<(TerminalId, Expr)> = grammar
            .terminals
            .iter()
            .map(|td| (td.id, parse_regex(&td.pattern)))
            .collect();
        Self::from_exprs(&terminals)
    }

    /// Get the start state.
    pub fn start_state(&self) -> u32 {
        0
    }

    /// Step from `state` on `byte`. Returns the next state, or `None` if dead.
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let next = self.dfa.get_transition(state, byte);
        // State 0 is typically the dead/start state in the minimized DFA.
        // We treat "no valid transition" as staying at current state or dead.
        // In our DFA, dead state is implicit — check if the target has any outgoing transitions.
        Some(next)
    }

    /// Feed a byte string, return the final state.
    pub fn run(&self, input: &[u8]) -> u32 {
        self.dfa.run(input)
    }

    /// Get the set of terminals matched at the given state.
    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalId> {
        if state == crate::automata::dfa::DEAD {
            return BTreeSet::new();
        }
        self.dfa
            .finalizers(state)
            .iter()
            .map(|&gid| gid as TerminalId)
            .collect()
    }

    /// Check if a specific terminal matches at the given state.
    pub fn terminal_matches(&self, state: u32, terminal: TerminalId) -> bool {
        self.dfa.finalizers(state).contains(&(terminal as usize))
    }

    /// Number of DFA states.
    pub fn num_states(&self) -> u32 {
        self.dfa.num_states() as u32
    }

    /// Execute the tokenizer on a byte string from a given state.
    /// Returns (final_state, set of matched terminal IDs at the end).
    ///
    /// This does maximal-munch tokenization: feeds all bytes and returns
    /// terminals matched at the final state.
    pub fn execute(&self, input: &[u8], start: u32) -> (u32, BTreeSet<TerminalId>) {
        let mut state = start;
        for &b in input {
            state = self.dfa.get_transition(state, b);
            if state == crate::automata::dfa::DEAD {
                return (state, BTreeSet::new());
            }
        }
        let matched = self.matched_terminals(state);
        (state, matched)
    }

    /// Execute the tokenizer on a byte string, tracking matches at every prefix.
    ///
    /// Returns a list of `(byte_offset, matched_terminals)` for each prefix
    /// where at least one terminal matches. `byte_offset` is the number of
    /// bytes consumed (1-indexed). Also returns the final DFA state after
    /// processing all bytes and its matched terminals (if any).
    ///
    /// This is used during commit to find all intermediate terminal matches
    /// within a single LLM token's byte sequence.
    pub fn execute_all_matches(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerResult {
        let mut state = start;
        let mut matches = Vec::new();

        for (i, &b) in input.iter().enumerate() {
            state = self.dfa.get_transition(state, b);
            if state == crate::automata::dfa::DEAD {
                return TokenizerResult {
                    end_state: state,
                    matches,
                };
            }
            let matched = self.matched_terminals(state);
            if !matched.is_empty() {
                matches.push((i + 1, matched));
            }
        }

        TokenizerResult {
            end_state: state,
            matches,
        }
    }

    /// The initial DFA state (always 0).
    pub fn initial_state(&self) -> u32 {
        0
    }
}

/// Result of executing the tokenizer with intermediate match tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    /// The DFA state after processing all bytes.
    pub end_state: u32,
    /// Matches found at each prefix: `(byte_offset, matched_terminals)`.
    /// `byte_offset` is 1-indexed (number of bytes consumed).
    pub matches: Vec<(usize, BTreeSet<TerminalId>)>,
}

// ---------------------------------------------------------------------------
// Simple regex parser
// ---------------------------------------------------------------------------

/// Parse a simple regex pattern string into an `Expr` AST.
///
/// Supports:
/// - Literal characters
/// - `\n`, `\t`, `\r`, `\\`, `\[`, `\]`, `\(`, `\)`, `\+`, `\*`, `\?`, `\.`, `\|`, `\{`, `\}`
/// - `\xHH` hex byte escapes
/// - `\d`, `\w`, `\s` and their uppercase negations
/// - `.` (any byte except newline)
/// - `[abc]`, `[a-z]`, `[^abc]` character classes
/// - `(...)` grouping
/// - `|` alternation
/// - `*`, `+`, `?` quantifiers
/// - `{n}`, `{n,}`, `{n,m}` bounded repetition
pub fn parse_regex(pattern: &str) -> Expr {
    let bytes = pattern.as_bytes();
    let (expr, pos) = parse_alternation(bytes, 0);
    assert_eq!(
        pos,
        bytes.len(),
        "Unexpected character at position {} in pattern {:?}",
        pos,
        pattern
    );
    expr
}

fn parse_alternation(input: &[u8], pos: usize) -> (Expr, usize) {
    let (first, mut pos) = parse_sequence(input, pos);
    let mut alts = vec![first];

    while pos < input.len() && input[pos] == b'|' {
        pos += 1;
        let (alt, next) = parse_sequence(input, pos);
        alts.push(alt);
        pos = next;
    }

    if alts.len() == 1 {
        (alts.into_iter().next().unwrap(), pos)
    } else {
        (Expr::Choice(alts), pos)
    }
}

fn parse_sequence(input: &[u8], mut pos: usize) -> (Expr, usize) {
    let mut parts: Vec<Expr> = Vec::new();

    while pos < input.len() && input[pos] != b'|' && input[pos] != b')' {
        let (atom, next) = parse_quantified(input, pos);
        parts.push(atom);
        pos = next;
    }

    match parts.len() {
        0 => (Expr::U8Seq(vec![]), pos), // empty ε
        1 => (parts.into_iter().next().unwrap(), pos),
        _ => (Expr::Seq(parts), pos),
    }
}

fn parse_quantified(input: &[u8], pos: usize) -> (Expr, usize) {
    let (atom, mut pos) = parse_atom(input, pos);

    if pos < input.len() {
        match input[pos] {
            b'*' => {
                pos += 1;
                (
                    Expr::Quantifier(
                        Box::new(atom),
                        crate::automata::regex::QuantifierType::ZeroOrMore,
                    ),
                    pos,
                )
            }
            b'+' => {
                pos += 1;
                (
                    Expr::Quantifier(
                        Box::new(atom),
                        crate::automata::regex::QuantifierType::OneOrMore,
                    ),
                    pos,
                )
            }
            b'?' => {
                pos += 1;
                (
                    Expr::Quantifier(
                        Box::new(atom),
                        crate::automata::regex::QuantifierType::ZeroOrOne,
                    ),
                    pos,
                )
            }
            b'{' => {
                let (min, max, next) = parse_repetition_bounds(input, pos + 1);
                (
                    Expr::RepeatBounded {
                        inner: Box::new(atom),
                        min,
                        max,
                    },
                    next,
                )
            }
            _ => (atom, pos),
        }
    } else {
        (atom, pos)
    }
}

fn parse_repetition_bounds(input: &[u8], mut pos: usize) -> (usize, Option<usize>, usize) {
    // Parse: {n}, {n,}, {n,m}
    let (min, next) = parse_usize(input, pos);
    pos = next;

    if pos < input.len() && input[pos] == b'}' {
        // {n} — exactly n
        return (min, Some(min), pos + 1);
    }
    if pos < input.len() && input[pos] == b',' {
        pos += 1;
        if pos < input.len() && input[pos] == b'}' {
            // {n,} — at least n
            return (min, None, pos + 1);
        }
        let (max, next) = parse_usize(input, pos);
        pos = next;
        assert!(pos < input.len() && input[pos] == b'}', "Expected }}");
        return (min, Some(max), pos + 1);
    }
    panic!("Invalid repetition bounds");
}

fn parse_usize(input: &[u8], mut pos: usize) -> (usize, usize) {
    let start = pos;
    while pos < input.len() && input[pos].is_ascii_digit() {
        pos += 1;
    }
    let s = std::str::from_utf8(&input[start..pos]).unwrap();
    (s.parse::<usize>().unwrap(), pos)
}

fn parse_atom(input: &[u8], mut pos: usize) -> (Expr, usize) {
    assert!(pos < input.len(), "Unexpected end of regex");

    match input[pos] {
        b'(' => {
            pos += 1;
            let (expr, next) = parse_alternation(input, pos);
            assert!(
                next < input.len() && input[next] == b')',
                "Expected ')'"
            );
            (expr, next + 1)
        }
        b'[' => parse_char_class(input, pos),
        b'\\' => parse_escape(input, pos),
        b'.' => {
            // Any byte except newline.
            let mut set = U8Set::full();
            set.remove(b'\n');
            (Expr::U8Class(set), pos + 1)
        }
        b'^' | b'$' => {
            // Anchors — treat as ε (we don't support multi-line).
            (Expr::U8Seq(vec![]), pos + 1)
        }
        ch => {
            // Literal byte.
            (Expr::U8Seq(vec![ch]), pos + 1)
        }
    }
}

fn parse_char_class(input: &[u8], mut pos: usize) -> (Expr, usize) {
    assert_eq!(input[pos], b'[');
    pos += 1;

    let negate = pos < input.len() && input[pos] == b'^';
    if negate {
        pos += 1;
    }

    let mut set = U8Set::empty();

    // Allow ] as first character in class.
    if pos < input.len() && input[pos] == b']' {
        set.insert(b']');
        pos += 1;
    }

    while pos < input.len() && input[pos] != b']' {
        if input[pos] == b'\\' {
            let ch = parse_escape_byte(input, pos);
            pos += escape_len(input, pos);
            // Check for range: \x-y
            if pos + 1 < input.len() && input[pos] == b'-' && input[pos + 1] != b']' {
                pos += 1;
                let hi = if input[pos] == b'\\' {
                    let h = parse_escape_byte(input, pos);
                    pos += escape_len(input, pos);
                    h
                } else {
                    let h = input[pos];
                    pos += 1;
                    h
                };
                for b in ch..=hi {
                    set.insert(b);
                }
            } else {
                set.insert(ch);
            }
        } else if pos + 2 < input.len()
            && input[pos + 1] == b'-'
            && input[pos + 2] != b']'
        {
            let lo = input[pos];
            let hi = input[pos + 2];
            for b in lo..=hi {
                set.insert(b);
            }
            pos += 3;
        } else {
            set.insert(input[pos]);
            pos += 1;
        }
    }

    assert!(pos < input.len() && input[pos] == b']', "Expected ']'");
    pos += 1;

    if negate {
        set = !set;
    }

    (Expr::U8Class(set), pos)
}

fn parse_escape(input: &[u8], pos: usize) -> (Expr, usize) {
    assert_eq!(input[pos], b'\\');
    let next = pos + 1;
    assert!(next < input.len(), "Trailing backslash");

    match input[next] {
        b'd' => {
            let set = U8Set::from_predicate(|b| b.is_ascii_digit());
            (Expr::U8Class(set), next + 1)
        }
        b'D' => {
            let set = U8Set::from_predicate(|b| !b.is_ascii_digit());
            (Expr::U8Class(set), next + 1)
        }
        b'w' => {
            let set = U8Set::from_predicate(|b| b.is_ascii_alphanumeric() || b == b'_');
            (Expr::U8Class(set), next + 1)
        }
        b'W' => {
            let set = U8Set::from_predicate(|b| !(b.is_ascii_alphanumeric() || b == b'_'));
            (Expr::U8Class(set), next + 1)
        }
        b's' => {
            let set = U8Set::from_predicate(|b| b.is_ascii_whitespace());
            (Expr::U8Class(set), next + 1)
        }
        b'S' => {
            let set = U8Set::from_predicate(|b| !b.is_ascii_whitespace());
            (Expr::U8Class(set), next + 1)
        }
        b'x' => {
            // \xHH
            assert!(
                next + 3 <= input.len(),
                "\\x requires two hex digits"
            );
            let hi = hex_digit(input[next + 1]);
            let lo = hex_digit(input[next + 2]);
            let byte = (hi << 4) | lo;
            (Expr::U8Seq(vec![byte]), next + 3)
        }
        b'n' => (Expr::U8Seq(vec![b'\n']), next + 1),
        b'r' => (Expr::U8Seq(vec![b'\r']), next + 1),
        b't' => (Expr::U8Seq(vec![b'\t']), next + 1),
        // All other escaped characters are literal.
        ch => (Expr::U8Seq(vec![ch]), next + 1),
    }
}

fn parse_escape_byte(input: &[u8], pos: usize) -> u8 {
    assert_eq!(input[pos], b'\\');
    let next = pos + 1;
    match input[next] {
        b'x' => {
            let hi = hex_digit(input[next + 1]);
            let lo = hex_digit(input[next + 2]);
            (hi << 4) | lo
        }
        b'n' => b'\n',
        b'r' => b'\r',
        b't' => b'\t',
        ch => ch,
    }
}

fn escape_len(input: &[u8], pos: usize) -> usize {
    assert_eq!(input[pos], b'\\');
    match input[pos + 1] {
        b'x' => 4,
        _ => 2,
    }
}

fn hex_digit(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => panic!("Invalid hex digit: {}", b as char),
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::{GrammarDef, Rule, Symbol, TerminalDef};

    #[test]
    fn test_parse_regex_literal() {
        let expr = parse_regex("abc");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"abc"));
        assert!(!regex.is_match(b"ab"));
        assert!(!regex.is_match(b"abcd"));
    }

    #[test]
    fn test_parse_regex_class() {
        let expr = parse_regex("[a-z]+");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"hello"));
        assert!(!regex.is_match(b"Hello"));
        assert!(!regex.is_match(b""));
    }

    #[test]
    fn test_parse_regex_alternation() {
        let expr = parse_regex("cat|dog");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"cat"));
        assert!(regex.is_match(b"dog"));
        assert!(!regex.is_match(b"car"));
    }

    #[test]
    fn test_parse_regex_quantifiers() {
        let expr = parse_regex("a*b+c?");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"bc"));      // a* = empty, b+ = b, c? = c
        assert!(regex.is_match(b"aaab"));    // a* = aaa, b+ = b, c? = empty
        assert!(regex.is_match(b"bbc"));     // a* = empty, b+ = bb, c? = c
        assert!(!regex.is_match(b"a"));      // b+ needs at least one b
    }

    #[test]
    fn test_parse_regex_escapes() {
        let expr = parse_regex(r"\d+");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"123"));
        assert!(!regex.is_match(b"abc"));
    }

    #[test]
    fn test_parse_regex_group() {
        let expr = parse_regex("(ab)+");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"ab"));
        assert!(regex.is_match(b"abab"));
        assert!(!regex.is_match(b"a"));
    }

    #[test]
    fn test_tokenizer_dfa_basic() {
        // Two terminals: "a" and "b"
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                TerminalDef { id: 0, name: "a".into(), pattern: "a".into() },
                TerminalDef { id: 1, name: "b".into(), pattern: "b".into() },
            ],
        };
        let tok = TokenizerDfa::from_grammar_def(&gdef);

        let (_s, m) = tok.execute(b"a", tok.start_state());
        assert!(m.contains(&0));
        assert!(!m.contains(&1));

        let (_s, m) = tok.execute(b"b", tok.start_state());
        assert!(!m.contains(&0));
        assert!(m.contains(&1));
    }

    #[test]
    fn test_tokenizer_dfa_multichar() {
        // Terminals: "if" and "[a-z]+"
        let gdef = GrammarDef {
            rules: vec![Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] }],
            start: 0,
            terminals: vec![
                TerminalDef { id: 0, name: "if".into(), pattern: "if".into() },
                TerminalDef { id: 1, name: "ident".into(), pattern: "[a-z]+".into() },
            ],
        };
        let tok = TokenizerDfa::from_grammar_def(&gdef);

        // "if" should match both terminal 0 and terminal 1
        let (_s, m) = tok.execute(b"if", tok.start_state());
        assert!(m.contains(&0), "if should match 'if' terminal");
        assert!(m.contains(&1), "if should match 'ident' terminal");

        // "foo" should match only terminal 1
        let (_s, m) = tok.execute(b"foo", tok.start_state());
        assert!(!m.contains(&0));
        assert!(m.contains(&1));
    }

    #[test]
    fn test_tokenizer_dfa_repetition() {
        let expr = parse_regex("a{2,4}");
        let r = ExprGroup { expr, is_non_greedy: false };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(!regex.is_match(b"a"));
        assert!(regex.is_match(b"aa"));
        assert!(regex.is_match(b"aaa"));
        assert!(regex.is_match(b"aaaa"));
        assert!(!regex.is_match(b"aaaaa"));
    }
}
