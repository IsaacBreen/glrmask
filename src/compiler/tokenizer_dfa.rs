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
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerDfa {
    /// The underlying multi-group DFA.
    pub dfa: Dfa,
    /// Number of terminals.
    pub num_terminals: u32,
}

impl TokenizerDfa {
    /// Build a tokenizer DFA from fully specified regex groups.
    pub fn from_expr_groups(groups: &[ExprGroup]) -> Self {
        let num_terminals = groups.len() as u32;
        let dfa = ExprGroups {
            groups: groups.to_vec(),
        }
        .build();
        Self {
            dfa: dfa.dfa,
            num_terminals,
        }
    }

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
        Self::from_expr_groups(&groups)
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

    /// Get the subset of matched terminals whose regex groups are marked non-greedy.
    pub fn matched_non_greedy_terminals(&self, state: u32) -> BTreeSet<TerminalId> {
        if state == crate::automata::dfa::DEAD {
            return BTreeSet::new();
        }
        self.dfa
            .non_greedy_finalizers(state)
            .iter()
            .map(|&gid| gid as TerminalId)
            .collect()
    }

    /// Get terminals that remain reachable on some non-empty continuation.
    pub fn possible_future_terminals(&self, state: u32) -> BTreeSet<TerminalId> {
        if state == crate::automata::dfa::DEAD {
            return BTreeSet::new();
        }
        self.dfa
            .possible_future_group_ids(state)
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

    /// Compute which terminals are reachable from each DFA state.
    ///
    /// `reachable_terminals[state]` = set of terminal IDs that can be reached
    /// (via some sequence of bytes) from `state`.
    ///
    /// This is a backward reachability analysis: start from accepting states,
    /// then propagate backward through transitions.
    pub fn compute_reachable_terminals(&self) -> Vec<BTreeSet<TerminalId>> {
        let n = self.num_states() as usize;
        let mut reachable: Vec<BTreeSet<TerminalId>> = vec![BTreeSet::new(); n];

        // Seed: each accepting state reaches its matched terminals.
        for s in 0..n {
            let s32 = s as u32;
            if s32 == crate::automata::dfa::DEAD {
                continue;
            }
            for &gid in self.dfa.finalizers(s32) {
                reachable[s].insert(gid as TerminalId);
            }
        }

        // Fixed-point backward propagation.
        let mut changed = true;
        while changed {
            changed = false;
            for s in 0..n {
                let s32 = s as u32;
                if s32 == crate::automata::dfa::DEAD {
                    continue;
                }
                for byte in 0..=255u8 {
                    let next = self.dfa.get_transition(s32, byte);
                    if next == crate::automata::dfa::DEAD || next as usize >= n {
                        continue;
                    }
                    // If next can reach terminal T, then s can reach T too.
                    let next_reachable = reachable[next as usize].clone();
                    for t in next_reachable {
                        if reachable[s].insert(t) {
                            changed = true;
                        }
                    }
                }
            }
        }

        reachable
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
    pub fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
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
            assert!(next < input.len() && input[next] == b')', "Expected ')'");
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
        } else if pos + 2 < input.len() && input[pos + 1] == b'-' && input[pos + 2] != b']' {
            let lo = input[pos];
            pos += 2; // skip lo and '-'
            let hi = if input[pos] == b'\\' {
                let h = parse_escape_byte(input, pos);
                pos += escape_len(input, pos);
                h
            } else {
                let h = input[pos];
                pos += 1;
                h
            };
            for b in lo..=hi {
                set.insert(b);
            }
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
            assert!(next + 3 <= input.len(), "\\x requires two hex digits");
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
    use crate::automata::regex::bytes;
    use crate::compiler::grammar_def::{GrammarDef, Rule, Symbol, TerminalDef};

    #[test]
    fn test_parse_regex_literal() {
        let expr = parse_regex("abc");
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"abc"));
        assert!(!regex.is_match(b"ab"));
        assert!(!regex.is_match(b"abcd"));
    }

    #[test]
    fn test_parse_regex_class() {
        let expr = parse_regex("[a-z]+");
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"hello"));
        assert!(!regex.is_match(b"Hello"));
        assert!(!regex.is_match(b""));
    }

    #[test]
    fn test_parse_regex_alternation() {
        let expr = parse_regex("cat|dog");
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"cat"));
        assert!(regex.is_match(b"dog"));
        assert!(!regex.is_match(b"car"));
    }

    #[test]
    fn test_parse_regex_quantifiers() {
        let expr = parse_regex("a*b+c?");
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"bc")); // a* = empty, b+ = b, c? = c
        assert!(regex.is_match(b"aaab")); // a* = aaa, b+ = b, c? = empty
        assert!(regex.is_match(b"bbc")); // a* = empty, b+ = bb, c? = c
        assert!(!regex.is_match(b"a")); // b+ needs at least one b
    }

    #[test]
    fn test_parse_regex_escapes() {
        let expr = parse_regex(r"\d+");
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(regex.is_match(b"123"));
        assert!(!regex.is_match(b"abc"));
    }

    #[test]
    fn test_parse_regex_group() {
        let expr = parse_regex("(ab)+");
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
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
                TerminalDef {
                    id: 0,
                    name: "a".into(),
                    pattern: "a".into(),
                },
                TerminalDef {
                    id: 1,
                    name: "b".into(),
                    pattern: "b".into(),
                },
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
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![
                TerminalDef {
                    id: 0,
                    name: "if".into(),
                    pattern: "if".into(),
                },
                TerminalDef {
                    id: 1,
                    name: "ident".into(),
                    pattern: "[a-z]+".into(),
                },
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
        let r = ExprGroup {
            expr,
            is_non_greedy: false,
        };
        let regex = ExprGroups { groups: vec![r] }.build();
        assert!(!regex.is_match(b"a"));
        assert!(regex.is_match(b"aa"));
        assert!(regex.is_match(b"aaa"));
        assert!(regex.is_match(b"aaaa"));
        assert!(!regex.is_match(b"aaaaa"));
    }

    #[test]
    fn test_reachable_terminals_boolean() {
        use crate::frontend::lark::parse_lark;

        let lark = "JSON_BOOL: \"true\" | \"false\"\nstart: JSON_BOOL\n";
        let grammar = parse_lark(lark).unwrap();
        let terminals: Vec<_> = grammar.terminals.iter().map(|td| (td.id, parse_regex(&td.pattern))).collect();
        eprintln!("Terminals: {:?}", terminals.iter().map(|(id, _)| *id).collect::<Vec<_>>());
        let tokenizer = TokenizerDfa::from_exprs(&terminals);
        eprintln!("DFA states: {}", tokenizer.dfa.num_states());

        // Check reachable terminals
        let reachable = tokenizer.compute_reachable_terminals();
        for (s, r) in reachable.iter().enumerate() {
            if !r.is_empty() {
                eprintln!("  State {}: reachable {:?}", s, r);
            }
        }

        let start = tokenizer.start_state();
        eprintln!("Start state: {}", start);

        // execute("t")
        let (es_t, m_t) = tokenizer.execute(b"t", start);
        eprintln!("execute(b\"t\") => state={}, matched={:?}", es_t, m_t);
        if es_t != crate::automata::dfa::DEAD {
            eprintln!("  reachable: {:?}", reachable.get(es_t as usize));
        }

        // execute("tr")
        let (es_tr, m_tr) = tokenizer.execute(b"tr", start);
        eprintln!("execute(b\"tr\") => state={}, matched={:?}", es_tr, m_tr);
        if es_tr != crate::automata::dfa::DEAD {
            eprintln!("  reachable: {:?}", reachable.get(es_tr as usize));
        }

        // execute("true")
        let (es_true, m_true) = tokenizer.execute(b"true", start);
        eprintln!("execute(b\"true\") => state={}, matched={:?}", es_true, m_true);

        // execute("f")
        let (es_f, m_f) = tokenizer.execute(b"f", start);
        eprintln!("execute(b\"f\") => state={}, matched={:?}", es_f, m_f);
        if es_f != crate::automata::dfa::DEAD {
            eprintln!("  reachable: {:?}", reachable.get(es_f as usize));
        }

        // The start state should be able to reach JSON_BOOL
        assert!(reachable[start as usize].contains(&0), "JSON_BOOL should be reachable from start");
        // After "t", JSON_BOOL should still be reachable
        assert_ne!(es_t, crate::automata::dfa::DEAD, "t should not be dead");
        assert!(
            reachable[es_t as usize].contains(&0),
            "JSON_BOOL should be reachable from state after 't'"
        );
    }

    #[test]
    fn test_preserves_non_greedy_and_future_terminal_metadata() {
        let tokenizer = TokenizerDfa::from_expr_groups(&[
            ExprGroup {
                expr: bytes(b"a"),
                is_non_greedy: true,
            },
            ExprGroup {
                expr: bytes(b"ab"),
                is_non_greedy: false,
            },
        ]);

        let state_after_a = tokenizer.run(b"a");
        assert!(tokenizer.matched_terminals(state_after_a).contains(&0));
        assert!(tokenizer
            .matched_non_greedy_terminals(state_after_a)
            .contains(&0));
        assert!(tokenizer
            .possible_future_terminals(state_after_a)
            .contains(&1));
    }
}
