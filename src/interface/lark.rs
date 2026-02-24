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
use std::sync::OnceLock;

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


fn parse_braced_repeat_suffix(suffix: &str) -> Option<Result<(usize, Option<usize>), String>> {
    if !(suffix.starts_with('{') && suffix.ends_with('}')) {
        return None;
    }
    let inner = &suffix[1..suffix.len() - 1];
    if inner.is_empty() {
        return Some(Err("Empty bounded-repeat suffix in regex literal".to_string()));
    }

    let parse_usize = |text: &str| -> Result<usize, String> {
        text.parse::<usize>()
            .map_err(|_| format!("Invalid repeat bound '{}' in regex literal", text))
    };

    if let Some((lhs, rhs)) = inner.split_once(',') {
        let min = match parse_usize(lhs.trim()) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let max = if rhs.trim().is_empty() {
            None
        } else {
            match parse_usize(rhs.trim()) {
                Ok(v) => Some(v),
                Err(e) => return Some(Err(e)),
            }
        };
        if let Some(max_val) = max {
            if max_val < min {
                return Some(Err(format!(
                    "Regex literal repeat upper bound {} is less than lower bound {}",
                    max_val, min
                )));
            }
        }
        return Some(Ok((min, max)));
    }

    match parse_usize(inner.trim()) {
        Ok(exact) => Some(Ok((exact, Some(exact)))),
        Err(e) => Some(Err(e)),
    }
}

fn literal_expr_for_char(c: char) -> GrammarExpr {
    let mut buf = [0_u8; 4];
    let bytes = c.encode_utf8(&mut buf).as_bytes().to_vec();
    GrammarExpr::Literal(bytes)
}

struct RegexLiteralParser<'a> {
    source: &'a str,
    pos: usize,
}

impl<'a> RegexLiteralParser<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, pos: 0 }
    }

    fn parse(mut self) -> Result<GrammarExpr, String> {
        let expr = self.parse_alternation()?;
        if self.peek_char().is_some() {
            return Err(format!(
                "Unexpected trailing regex content '{}' in '/{}/'",
                &self.source[self.pos..],
                self.source
            ));
        }
        Ok(expr)
    }

    fn peek_char(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn parse_alternation(&mut self) -> Result<GrammarExpr, String> {
        let mut branches = vec![self.parse_concatenation()?];
        while self.peek_char() == Some('|') {
            self.bump_char();
            branches.push(self.parse_concatenation()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(GrammarExpr::Choice(branches))
        }
    }

    fn parse_concatenation(&mut self) -> Result<GrammarExpr, String> {
        let mut parts: Vec<GrammarExpr> = Vec::new();
        while let Some(ch) = self.peek_char() {
            if ch == ')' || ch == '|' {
                break;
            }
            let term = self.parse_repetition()?;
            if !matches!(term, GrammarExpr::Sequence(ref items) if items.is_empty()) {
                parts.push(term);
            }
        }

        if parts.is_empty() {
            Ok(GrammarExpr::Sequence(vec![]))
        } else if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            Ok(GrammarExpr::Sequence(parts))
        }
    }

    fn parse_repetition(&mut self) -> Result<GrammarExpr, String> {
        let atom = self.parse_atom()?;

        match self.peek_char() {
            Some('?') => {
                self.bump_char();
                Ok(GrammarExpr::Optional(Box::new(atom)))
            }
            Some('*') => {
                self.bump_char();
                Ok(GrammarExpr::Repeat(Box::new(atom)))
            }
            Some('+') => {
                self.bump_char();
                Ok(GrammarExpr::Sequence(vec![
                    atom.clone(),
                    GrammarExpr::Repeat(Box::new(atom)),
                ]))
            }
            Some('{') => {
                let start = self.pos;
                self.bump_char();
                while let Some(ch) = self.peek_char() {
                    self.bump_char();
                    if ch == '}' {
                        break;
                    }
                }
                let suffix = &self.source[start..self.pos];
                let (min, max) = parse_braced_repeat_suffix(suffix)
                    .ok_or_else(|| format!("Invalid bounded repeat suffix '{}'", suffix))??;
                Ok(GrammarExpr::RepeatBounded {
                    min,
                    max,
                    inner: Box::new(atom),
                })
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<GrammarExpr, String> {
        match self.peek_char() {
            Some('(') => {
                self.bump_char();
                let inner = self.parse_alternation()?;
                if self.peek_char() != Some(')') {
                    return Err(format!("Unclosed group in regex '/{}/'", self.source));
                }
                self.bump_char();
                Ok(inner)
            }
            Some('[') => self.parse_char_class(),
            Some('.') => {
                self.bump_char();
                Ok(GrammarExpr::AnyChar)
            }
            Some('^') | Some('$') => {
                // Anchors are zero-width assertions in regex syntax.
                self.bump_char();
                Ok(GrammarExpr::Sequence(vec![]))
            }
            Some('\\') => self.parse_escape(),
            Some(ch) => {
                self.bump_char();
                Ok(literal_expr_for_char(ch))
            }
            None => Err("Unexpected end of regex literal".to_string()),
        }
    }

    fn parse_char_class(&mut self) -> Result<GrammarExpr, String> {
        let start = self.pos;
        self.bump_char(); // consume '['

        let mut escaped = false;
        while let Some(ch) = self.bump_char() {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == ']' {
                let def = self.source[start..self.pos].to_string();
                return Ok(GrammarExpr::CharClass { def, utf8: true });
            }
        }

        Err(format!("Unterminated character class in regex '/{}/'", self.source))
    }

    fn parse_escape(&mut self) -> Result<GrammarExpr, String> {
        self.bump_char(); // consume '\\'
        let escaped = self
            .bump_char()
            .ok_or_else(|| "Dangling escape in regex literal".to_string())?;

        let class_expr = |def: &str| GrammarExpr::CharClass {
            def: def.to_string(),
            utf8: true,
        };

        match escaped {
            'd' => Ok(class_expr("[0-9]")),
            'D' => Ok(class_expr("[^0-9]")),
            'w' => Ok(class_expr("[A-Za-z0-9_]")),
            'W' => Ok(class_expr("[^A-Za-z0-9_]")),
            's' => Ok(class_expr("[\\t\\n\\r\\f\\v ]")),
            'S' => Ok(class_expr("[^\\t\\n\\r\\f\\v ]")),
            'n' => Ok(literal_expr_for_char('\n')),
            't' => Ok(literal_expr_for_char('\t')),
            'r' => Ok(literal_expr_for_char('\r')),
            'x' => {
                let h1 = self
                    .bump_char()
                    .ok_or_else(|| "Incomplete \\xNN escape in regex literal".to_string())?;
                let h2 = self
                    .bump_char()
                    .ok_or_else(|| "Incomplete \\xNN escape in regex literal".to_string())?;
                let hex = format!("{}{}", h1, h2);
                let byte = u8::from_str_radix(&hex, 16)
                    .map_err(|_| format!("Invalid \\x{} escape in regex literal", hex))?;
                Ok(GrammarExpr::Literal(vec![byte]))
            }
            other => Ok(literal_expr_for_char(other)),
        }
    }
}

fn regex_literal_to_expr(regex_content: &str) -> Result<GrammarExpr, String> {
    RegexLiteralParser::new(regex_content).parse()
}


#[derive(Debug)]
pub(super) struct LarkParseResult {
    pub grammar_rules: Vec<(String, GrammarExpr)>,
    pub ignore_symbol_name: Option<String>,
}

pub(super) struct LarkParser<'a> {
    source: &'a str,
    tokens: Vec<LarkToken>,
    pos: usize,
}

impl<'a> LarkParser<'a> {
    pub(super) fn new(source: &'a str) -> Result<Self, LarkParseError> {
        let tokens = tokenize_lark(source)?;
        Ok(LarkParser {
            source,
            tokens,
            pos: 0,
        })
    }

    pub(super) fn parse(&mut self) -> Result<LarkParseResult, LarkParseError> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
        let mut seen_names = HashSet::new();
        let mut ignore_symbol_name = None;

        // Skip leading newlines
        self.skip_newlines();

        while self.peek().is_some() {
            if self.peek_directive("%ignore") {
                self.consume_directive("%ignore")?;
                let (symbol_name, _) = self.expect_ident()?;
                ignore_symbol_name = Some(symbol_name);
                self.skip_newlines();
            } else if self.peek_directive("%import") {
                // Skip %import directives - consume until newline
                self.advance();
                while self.peek().is_some() && !self.peek_newline() {
                    self.advance();
                }
                self.skip_newlines();
            } else if self.peek_directive_any() {
                // Skip unknown directives
                self.advance();
                while self.peek().is_some() && !self.peek_newline() {
                    self.advance();
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
                let saved_pos = self.pos;
                self.skip_newlines();
                if self.peek_op("|") {
                    self.consume_op("|")?;
                    self.skip_newlines();
                    choices.push(self.parse_sequence()?);
                } else {
                    // Not a continuation, restore position and break
                    self.pos = saved_pos;
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
        
        while self.peek().is_some()
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
        if let Some(LarkToken { kind: LarkTokenKind::Ident(id), .. }) = self.peek().cloned() {
            self.advance();
            Ok(r#ref(&id))
        } else if let Some(LarkToken { kind: LarkTokenKind::Literal(lit), .. }) = self.peek().cloned() {
            self.advance();
            Ok(literal(lit.into_bytes()))
        } else if let Some(LarkToken { kind: LarkTokenKind::CharClass(cc), .. }) = self.peek().cloned() {
            self.advance();
            Ok(GrammarExpr::CharClass {
                def: cc,
                utf8: false,
            })
        } else if let Some(LarkToken { kind: LarkTokenKind::RegexLiteral(re), span }) = self.peek().cloned() {
            self.advance();
            regex_literal_to_expr(&re)
                .map_err(|msg| LarkParseError::new(self.source, span, msg))
        } else if self.peek_op("(") {
            self.consume_op("(")?;
            self.skip_newlines();
            let expr = self.parse_expression()?;
            self.skip_newlines();
            self.expect_op(")")?;
            Ok(expr)
        } else {
            let (message, span) = if let Some(token) = self.peek() {
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
            self.advance();
        }
    }

    fn peek_newline(&self) -> bool {
        matches!(self.peek(), Some(LarkToken { kind: LarkTokenKind::Newline, .. }))
    }

    fn peek_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(LarkToken { kind: LarkTokenKind::Op(s), .. }) if s == op)
    }

    fn consume_op(&mut self, op: &str) -> Result<(), LarkParseError> {
        if self.peek_op(op) {
            self.advance();
            Ok(())
        } else {
            let (message, span) = if let Some(token) = self.peek() {
                (format!("Expected '{}', found {:?}", op, &token.kind), token.span)
            } else {
                (format!("Expected '{}', found end of input", op), self.eof_span())
            };
            Err(LarkParseError::new(self.source, span, message))
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<(), LarkParseError> {
        match self.next() {
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

    fn peek_directive(&self, directive: &str) -> bool {
        matches!(self.peek(), Some(LarkToken { kind: LarkTokenKind::Directive(s), .. }) if s == directive)
    }

    fn peek_directive_any(&self) -> bool {
        matches!(self.peek(), Some(LarkToken { kind: LarkTokenKind::Directive(_), .. }))
    }

    fn consume_directive(&mut self, directive: &str) -> Result<(), LarkParseError> {
        if self.peek_directive(directive) {
            self.advance();
            Ok(())
        } else {
            let (message, span) = if let Some(token) = self.peek() {
                (format!("Expected '{}', found {:?}", directive, &token.kind), token.span)
            } else {
                (format!("Expected '{}', found end of input", directive), self.eof_span())
            };
            Err(LarkParseError::new(self.source, span, message))
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), LarkParseError> {
        match self.next() {
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
        match self.next() {
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

    /// Peek at the current token without advancing.
    #[inline]
    fn peek(&self) -> Option<&LarkToken> {
        self.tokens.get(self.pos)
    }

    /// Peek at the token at offset `n` from the current position.
    #[inline]
    fn peek_nth(&self, n: usize) -> Option<&LarkToken> {
        self.tokens.get(self.pos + n)
    }

    /// Advance the position by one.
    #[inline]
    fn advance(&mut self) -> Option<()> {
        if self.pos < self.tokens.len() {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    /// Consume and return the next token (owned via clone).
    fn next(&mut self) -> Option<LarkToken> {
        if self.pos < self.tokens.len() {
            let token = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Check if we're at the start of a new rule definition (ident followed by :)
    /// O(1) — just index-based lookahead, no cloning.
    fn is_at_rule_start(&self) -> bool {
        matches!(
            (self.peek(), self.peek_nth(1)),
            (
                Some(LarkToken { kind: LarkTokenKind::Ident(_), .. }),
                Some(LarkToken { kind: LarkTokenKind::Op(op), .. })
            ) if op == ":"
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
