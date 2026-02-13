//! Lark grammar format parser.
//!
//! This module provides parsing for Lark-style grammars (used by lark-parser and llguidance).
//! Lark format uses `:` for rule definitions (instead of `::=`), no semicolons,
//! and supports `/regex/` patterns.
//!
//! Key differences from EBNF:
//! - Rule definition: `rule: expr` instead of `rule ::= expr;`
//! - Regex literals: `/[a-z]+/` instead of `[a-z]+`
//! - Directives: `%ignore TERMINAL` instead of `#![ignore(TERMINAL)]`
//! - Start rule: `start` is the default start rule name
//! - Terminals: UPPERCASE names are terminals (convention, not enforced)

use crate::interface::{choice, literal, optional, r#ref, repeat, repeat_bounded, sequence, GrammarExpr};
use regex::Regex;
use std::collections::HashSet;
use std::iter::Peekable;
use std::sync::OnceLock;
use std::vec::IntoIter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LarkParseError {
    message: String,
    span: Span,
    line: usize,
    column: usize,
    line_text: String,
}

impl LarkParseError {
    fn new(source: &str, span: Span, message: impl Into<String>) -> Self {
        let (line, column, line_text) = compute_line_info(source, span.start);
        LarkParseError {
            message: message.into(),
            span,
            line,
            column,
            line_text,
        }
    }
}

fn compute_line_info(source: &str, byte_index: usize) -> (usize, usize, String) {
    let len = source.len();
    let idx = byte_index.min(len);

    let mut line_start = 0;
    let mut line = 1;

    for (i, ch) in source.char_indices() {
        if i >= idx {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = i + 1;
        }
    }

    let line_end = source[line_start..]
        .find('\n')
        .map(|off| line_start + off)
        .unwrap_or(len);

    let line_text = source[line_start..line_end].to_string();
    let column = source[line_start..idx].chars().count() + 1;

    (line, column, line_text)
}

impl std::fmt::Display for LarkParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "error: Lark parse error: {}", self.message)?;
        write!(
            f,
            " --> line {}, column {} (byte range {}-{})",
            self.line, self.column, self.span.start, self.span.end
        )?;

        if !self.line_text.is_empty() {
            writeln!(f)?;
            writeln!(f, "  |")?;
            writeln!(f, "  | {}", self.line_text)?;
            write!(f, "  | ")?;
            let prefix: String = self
                .line_text
                .chars()
                .take(self.column.saturating_sub(1))
                .map(|c| if c == '\t' { '\t' } else { ' ' })
                .collect();
            write!(f, "{}", prefix)?;
            write!(f, "^")?;
        }

        Ok(())
    }
}

impl std::error::Error for LarkParseError {}

impl From<LarkParseError> for String {
    fn from(e: LarkParseError) -> Self {
        e.to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LarkTokenKind {
    Ident(String),
    Number(String),
    Literal(String),
    CharClass(String),  // [...] character class
    RegexLiteral(String), // /.../ regex
    Op(String),
    Directive(String),  // %ignore, %import, etc.
    Newline,
}

#[derive(Debug, Clone, PartialEq)]
struct LarkToken {
    kind: LarkTokenKind,
    span: Span,
}

fn get_lark_token_regex() -> &'static Regex {
    static LARK_TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
    LARK_TOKEN_REGEX.get_or_init(|| {
        // Note: Not using (?x) mode because it ignores unescaped spaces
        Regex::new(
            r#"(?P<ws>[ \t]+)|(?P<comment>//[^\r\n]*|#[^\r\n]*)|(?P<directive>%[a-zA-Z_]+)|(?P<regex>/([^/\\]|\\.)+/)|(?P<number>[0-9]+)|(?P<ident>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<literal>"([^"\\]|\\.)*"|'([^'\\]|\\.)*')|(?P<charclass>\[([^\]\[\(\)\{\}\\]|\\.)*\])|(?P<op>:|\?|\*|\+|\||\(|\)|\.\.|\~)|(?P<newline>\r?\n)|(?P<error>.)"#,
        )
        .unwrap()
    })
}

fn tokenize_lark(source: &str) -> Result<Vec<LarkToken>, LarkParseError> {
    let mut tokens = Vec::new();
    
    for cap in get_lark_token_regex().captures_iter(source) {
        if let Some(m) = cap.name("directive") {
            tokens.push(LarkToken {
                kind: LarkTokenKind::Directive(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("regex") {
            let s = m.as_str();
            let regex_content = &s[1..s.len() - 1]; // Strip the slashes
            tokens.push(LarkToken {
                kind: LarkTokenKind::RegexLiteral(regex_content.to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("number") {
            tokens.push(LarkToken {
                kind: LarkTokenKind::Number(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("ident") {
            tokens.push(LarkToken {
                kind: LarkTokenKind::Ident(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("literal") {
            let s = m.as_str();
            let content = &s[1..s.len() - 1];
            let unescaped = unescape_string_literal(content, source, m.start(), m.end())?;
            tokens.push(LarkToken {
                kind: LarkTokenKind::Literal(unescaped),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("charclass") {
            tokens.push(LarkToken {
                kind: LarkTokenKind::CharClass(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("op") {
            tokens.push(LarkToken {
                kind: LarkTokenKind::Op(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("newline") {
            tokens.push(LarkToken {
                kind: LarkTokenKind::Newline,
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(e) = cap.name("error") {
            let err_text = e.as_str();
            return Err(LarkParseError::new(
                source,
                Span { start: e.start(), end: e.end() },
                format!("Unknown token in Lark grammar: {:?}", err_text),
            ));
        }
        // ws and comment are ignored
    }
    Ok(tokens)
}

fn unescape_string_literal(content: &str, source: &str, start: usize, end: usize) -> Result<String, LarkParseError> {
    let mut unescaped = String::with_capacity(content.len());
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(escaped_char) = chars.next() {
                match escaped_char {
                    'n' => unescaped.push('\n'),
                    't' => unescaped.push('\t'),
                    'r' => unescaped.push('\r'),
                    'b' => unescaped.push('\u{0008}'),
                    'f' => unescaped.push('\u{000C}'),
                    'v' => unescaped.push('\u{000B}'),
                    '\\' => unescaped.push('\\'),
                    '\'' => unescaped.push('\''),
                    '"' => unescaped.push('"'),
                    '/' => unescaped.push('/'),
                    other => unescaped.push(other),
                }
            } else {
                return Err(LarkParseError::new(
                    source,
                    Span { start, end },
                    "Literal with dangling escape".to_string(),
                ));
            }
        } else {
            unescaped.push(c);
        }
    }
    Ok(unescaped)
}

#[derive(Debug)]
pub(super) struct LarkParseResult {
    pub grammar_rules: Vec<(String, GrammarExpr)>,
    pub ignore_symbol_name: Option<String>,
}

pub(super) struct LarkParser<'a> {
    source: &'a str,
    tokens: Peekable<IntoIter<LarkToken>>,
}

impl<'a> LarkParser<'a> {
    pub(super) fn new(source: &'a str) -> Result<Self, LarkParseError> {
        let tokens = tokenize_lark(source)?;
        Ok(LarkParser {
            source,
            tokens: tokens.into_iter().peekable(),
        })
    }

    pub(super) fn parse(&mut self) -> Result<LarkParseResult, LarkParseError> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
        let mut seen_names = HashSet::new();
        let mut ignore_symbol_name = None;

        // Skip leading newlines
        self.skip_newlines();

        while self.tokens.peek().is_some() {
            if self.peek_directive("%ignore") {
                self.consume_directive("%ignore")?;
                let (symbol_name, _) = self.expect_ident()?;
                ignore_symbol_name = Some(symbol_name);
                self.skip_newlines();
            } else if self.peek_directive("%import") {
                // Skip %import directives - consume until newline
                self.tokens.next();
                while self.tokens.peek().is_some() && !self.peek_newline() {
                    self.tokens.next();
                }
                self.skip_newlines();
            } else if self.peek_directive_any() {
                // Skip unknown directives
                self.tokens.next();
                while self.tokens.peek().is_some() && !self.peek_newline() {
                    self.tokens.next();
                }
                self.skip_newlines();
            } else if self.peek_newline() {
                self.skip_newlines();
            } else {
                let (rule_name, rule_name_span) = self.expect_ident()?;
                if seen_names.contains(&rule_name) {
                    return Err(LarkParseError::new(
                        self.source,
                        rule_name_span,
                        format!("Duplicate rule name: {}", rule_name),
                    ));
                }
                seen_names.insert(rule_name.clone());
                let rule_expr = self.parse_rule_body()?;
                rules.push((rule_name, rule_expr));
                self.skip_newlines();
            }
        }

        // Lark uses 'start' as the default start rule
        if let Some(start_idx) = rules.iter().position(|(name, _)| name == "start") {
            if start_idx > 0 {
                let start_rule = rules.remove(start_idx);
                rules.insert(0, start_rule);
            }
        }

        Ok(LarkParseResult {
            grammar_rules: rules,
            ignore_symbol_name,
        })
    }

    fn parse_rule_body(&mut self) -> Result<GrammarExpr, LarkParseError> {
        self.expect_op(":")?;
        let expr = self.parse_expression()?;
        Ok(expr)
    }

    fn parse_expression(&mut self) -> Result<GrammarExpr, LarkParseError> {
        let mut choices = vec![self.parse_sequence()?];
        
        loop {
            // Check for continuation: newline(s) followed by |
            // Or direct |
            if self.peek_op("|") {
                self.consume_op("|")?;
                self.skip_newlines();
                choices.push(self.parse_sequence()?);
            } else if self.peek_newline() {
                // Look ahead: is there a | after the newlines?
                let saved_pos = self.tokens.clone();
                self.skip_newlines();
                if self.peek_op("|") {
                    self.consume_op("|")?;
                    self.skip_newlines();
                    choices.push(self.parse_sequence()?);
                } else {
                    // Not a continuation, restore position and break
                    self.tokens = saved_pos;
                    break;
                }
            } else {
                break;
            }
        }
        
        if choices.len() == 1 {
            Ok(choices.remove(0))
        } else {
            Ok(choice(choices))
        }
    }

    fn parse_sequence(&mut self) -> Result<GrammarExpr, LarkParseError> {
        let mut terms = Vec::new();
        
        while self.tokens.peek().is_some()
            && !self.peek_op(")")
            && !self.peek_op("|")
            && !self.peek_newline()
            && !self.peek_directive_any()
        {
            // Check if we're at the start of a new rule (ident followed by :)
            if self.is_at_rule_start() {
                break;
            }
            terms.push(self.parse_term()?);
        }

        if terms.len() == 1 {
            Ok(terms.remove(0))
        } else {
            Ok(sequence(terms))
        }
    }

    fn parse_term(&mut self) -> Result<GrammarExpr, LarkParseError> {
        let factor = self.parse_factor()?;

        if self.peek_op("?") {
            self.consume_op("?")?;
            Ok(optional(factor))
        } else if self.peek_op("*") {
            self.consume_op("*")?;
            Ok(repeat(factor))
        } else if self.peek_op("+") {
            self.consume_op("+")?;
            Ok(sequence(vec![factor.clone(), repeat(factor)]))
        } else if self.peek_op("~") {
            // Lark repeat syntax: ~n or ~n..m
            self.consume_op("~")?;
            let min = self.parse_repeat_number()?;
            let max = if self.peek_op("..") {
                self.consume_op("..")?;
                Some(self.parse_repeat_number()?)
            } else {
                Some(min)
            };
            if let Some(max) = max {
                if max < min {
                    return Err(LarkParseError::new(
                        self.source,
                        self.eof_span(),
                        format!("Repeat upper bound {} is less than lower bound {}", max, min),
                    ));
                }
            }
            Ok(repeat_bounded(factor, min, max))
        } else {
            Ok(factor)
        }
    }

    fn parse_factor(&mut self) -> Result<GrammarExpr, LarkParseError> {
        if let Some(LarkToken { kind: LarkTokenKind::Ident(id), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(r#ref(&id))
        } else if let Some(LarkToken { kind: LarkTokenKind::Literal(lit), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(literal(lit.into_bytes()))
        } else if let Some(LarkToken { kind: LarkTokenKind::CharClass(cc), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(GrammarExpr::CharClass {
                def: cc,
                utf8: true,
            })
        } else if let Some(LarkToken { kind: LarkTokenKind::RegexLiteral(re), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            // Convert regex to character class format.
            // If the regex is already a single character class like /[^"\\]/,
            // preserve it verbatim to avoid producing nested brackets.
            if re.starts_with('[') && re.ends_with(']') {
                Ok(GrammarExpr::CharClass {
                    def: re,
                    utf8: true,
                })
            } else {
                Ok(GrammarExpr::CharClass {
                    def: format!("[{}]", re),
                    utf8: true,
                })
            }
        } else if self.peek_op("(") {
            self.consume_op("(")?;
            self.skip_newlines();
            let expr = self.parse_expression()?;
            self.skip_newlines();
            self.expect_op(")")?;
            Ok(expr)
        } else {
            let (message, span) = if let Some(token) = self.tokens.peek() {
                (
                    format!("Expected identifier, literal, or group, found {:?}", &token.kind),
                    token.span,
                )
            } else {
                (
                    "Expected identifier, literal, or group, found end of input".to_string(),
                    self.eof_span(),
                )
            };
            Err(LarkParseError::new(self.source, span, message))
        }
    }

    // --- Helper methods ---

    fn eof_span(&self) -> Span {
        let end = self.source.len();
        Span { start: end, end }
    }

    fn skip_newlines(&mut self) {
        while self.peek_newline() {
            self.tokens.next();
        }
    }

    fn peek_newline(&mut self) -> bool {
        matches!(self.tokens.peek(), Some(LarkToken { kind: LarkTokenKind::Newline, .. }))
    }

    fn peek_op(&mut self, op: &str) -> bool {
        matches!(self.tokens.peek(), Some(LarkToken { kind: LarkTokenKind::Op(s), .. }) if s == op)
    }

    fn consume_op(&mut self, op: &str) -> Result<(), LarkParseError> {
        if self.peek_op(op) {
            self.tokens.next();
            Ok(())
        } else {
            let (message, span) = if let Some(token) = self.tokens.peek() {
                (format!("Expected '{}', found {:?}", op, &token.kind), token.span)
            } else {
                (format!("Expected '{}', found end of input", op), self.eof_span())
            };
            Err(LarkParseError::new(self.source, span, message))
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<(), LarkParseError> {
        match self.tokens.next() {
            Some(LarkToken { kind: LarkTokenKind::Op(s), .. }) if s == op => Ok(()),
            Some(other) => Err(LarkParseError::new(
                self.source,
                other.span,
                format!("Expected '{}', found {:?}", op, other.kind),
            )),
            None => Err(LarkParseError::new(
                self.source,
                self.eof_span(),
                format!("Expected '{}', found end of input", op),
            )),
        }
    }

    fn peek_directive(&mut self, directive: &str) -> bool {
        matches!(self.tokens.peek(), Some(LarkToken { kind: LarkTokenKind::Directive(s), .. }) if s == directive)
    }

    fn peek_directive_any(&mut self) -> bool {
        matches!(self.tokens.peek(), Some(LarkToken { kind: LarkTokenKind::Directive(_), .. }))
    }

    fn consume_directive(&mut self, directive: &str) -> Result<(), LarkParseError> {
        if self.peek_directive(directive) {
            self.tokens.next();
            Ok(())
        } else {
            let (message, span) = if let Some(token) = self.tokens.peek() {
                (format!("Expected '{}', found {:?}", directive, &token.kind), token.span)
            } else {
                (format!("Expected '{}', found end of input", directive), self.eof_span())
            };
            Err(LarkParseError::new(self.source, span, message))
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), LarkParseError> {
        match self.tokens.next() {
            Some(LarkToken { kind: LarkTokenKind::Ident(id), span }) => Ok((id, span)),
            Some(other) => Err(LarkParseError::new(
                self.source,
                other.span,
                format!("Expected identifier, found {:?}", other.kind),
            )),
            None => Err(LarkParseError::new(
                self.source,
                self.eof_span(),
                "Expected identifier, found end of input".to_string(),
            )),
        }
    }

    fn parse_repeat_number(&mut self) -> Result<usize, LarkParseError> {
        match self.tokens.next() {
            Some(LarkToken { kind: LarkTokenKind::Number(num), span }) => num
                .parse::<usize>()
                .map_err(|_| LarkParseError::new(self.source, span, "Invalid repeat count")),
            Some(LarkToken { kind: LarkTokenKind::Ident(id), span }) => {
                if id.chars().all(|ch| ch.is_ascii_digit()) {
                    id.parse::<usize>()
                        .map_err(|_| LarkParseError::new(self.source, span, "Invalid repeat count"))
                } else {
                    Err(LarkParseError::new(
                        self.source,
                        span,
                        format!("Expected repeat count, found identifier {}", id),
                    ))
                }
            }
            Some(other) => Err(LarkParseError::new(
                self.source,
                other.span,
                format!("Expected repeat count, found {:?}", other.kind),
            )),
            None => Err(LarkParseError::new(
                self.source,
                self.eof_span(),
                "Expected repeat count, found end of input".to_string(),
            )),
        }
    }

    /// Check if we're at the start of a new rule definition (ident followed by :)
    fn is_at_rule_start(&mut self) -> bool {
        let tokens: Vec<_> = self.tokens.clone().take(2).collect();
        matches!(
            tokens.as_slice(),
            [
                LarkToken { kind: LarkTokenKind::Ident(_), .. },
                LarkToken { kind: LarkTokenKind::Op(op), .. }
            ] if op == ":"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lark_parser_simple() {
        let lark = r#"
start: expr

expr: term ("+" term)*

term: NUMBER
    | "(" expr ")"

NUMBER: /[0-9]+/
"#;
        let mut parser = LarkParser::new(lark).unwrap();
        let result = parser.parse().unwrap();

        assert_eq!(result.grammar_rules.len(), 4);
        assert_eq!(result.grammar_rules[0].0, "start");
        assert_eq!(result.grammar_rules[1].0, "expr");
        assert_eq!(result.grammar_rules[2].0, "term");
        assert_eq!(result.grammar_rules[3].0, "NUMBER");
    }

    #[test]
    fn test_lark_ignore_directive() {
        let lark = r#"
start: "hello"
WS: /\s+/
%ignore WS
"#;
        let mut parser = LarkParser::new(lark).unwrap();
        let result = parser.parse().unwrap();

        assert_eq!(result.ignore_symbol_name, Some("WS".to_string()));
    }

    #[test]
    fn test_lark_repeat_bounded() {
        let lark = r#"
start: STR_CHAR~3..5
STR_CHAR: "a"
"#;
        let mut parser = LarkParser::new(lark).unwrap();
        let result = parser.parse().unwrap();

        assert_eq!(result.grammar_rules[0].0, "start");
        assert_eq!(
            result.grammar_rules[0].1,
            GrammarExpr::RepeatBounded {
                min: 3,
                max: Some(5),
                inner: Box::new(GrammarExpr::Ref("STR_CHAR".to_string())),
            }
        );
    }

    #[test]
    fn test_lark_regex_charclass_not_nested() {
        let lark = r#"
start: STR_CHAR
STR_CHAR: /[^"\\\x00-\x1F]/
"#;
        let mut parser = LarkParser::new(lark).unwrap();
        let result = parser.parse().unwrap();

        let str_char_expr = result
            .grammar_rules
            .iter()
            .find(|(name, _)| name == "STR_CHAR")
            .map(|(_, expr)| expr)
            .expect("STR_CHAR rule should exist");

        match str_char_expr {
            GrammarExpr::CharClass { def: cc, .. } => {
                assert_eq!(cc, "[^\"\\\\\\x00-\\x1F]");
                assert!(!cc.starts_with("[["));
            }
            other => panic!("expected CharClass, got {:?}", other),
        }
    }
}
