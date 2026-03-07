//! Lark grammar parser.
//!
//! Parses Lark-format grammars into the internal `GrammarDef` IR.
//!
//! # Supported syntax
//!
//! ```text
//! // Comments start with //
//! rule_name: expression
//!
//! // Lowercase names are rules, UPPERCASE names are terminals.
//! // Terminals are defined with regex patterns:
//! TERMINAL: /regex/
//! TERMINAL: "literal"
//!
//! // Expressions:
//! a b c             // sequence
//! a | b | c         // choice
//! (expr)            // grouping
//! expr?             // optional
//! expr*             // zero or more
//! expr+             // one or more
//! "literal"         // literal string
//! /regex/           // regex terminal
//! rule_name         // rule reference
//! TERMINAL_NAME     // terminal reference
//! ```
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::frontend::ast::{GrammarExpr, NamedGrammar, lower};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),    // lowercase rule name
    Terminal(String), // UPPERCASE terminal name
    Literal(String),  // "string"
    Regex(String),    // /regex/
    LParen,
    RParen,
    LBracket,
    RBracket,
    Pipe,
    Star,
    Plus,
    Question,
    Colon,
    Newline,
    Dot,
    Tilde, // ~
    Number(usize),
    Comma,
    Arrow, // ->
}

struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.input.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace_inline(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(b) = self.peek() {
            if b == b'\n' {
                break;
            }
            self.pos += 1;
        }
    }

    fn lex_string(&mut self, quote: u8) -> Result<String, GlrMaskError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(b) if b == quote => return Ok(s),
                Some(b'\\') => match self.advance() {
                    Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'),
                    Some(b'r') => s.push('\r'),
                    Some(b'\\') => s.push('\\'),
                    Some(b'"') => s.push('"'),
                    Some(b'x') => {
                        // Hex escape: \xHH
                        let h1 = self.advance().ok_or_else(|| {
                            GlrMaskError::GrammarParse("unterminated \\x escape".into())
                        })?;
                        let h2 = self.advance().ok_or_else(|| {
                            GlrMaskError::GrammarParse("unterminated \\x escape".into())
                        })?;
                        let hex_str = format!("{}{}", h1 as char, h2 as char);
                        let byte = u8::from_str_radix(&hex_str, 16).map_err(|_| {
                            GlrMaskError::GrammarParse(format!("invalid \\x escape: \\x{hex_str}"))
                        })?;
                        s.push(byte as char);
                    }
                    Some(c) => {
                        s.push('\\');
                        s.push(c as char);
                    }
                    None => return Err(GlrMaskError::GrammarParse("unterminated escape".into())),
                },
                Some(b) => s.push(b as char),
                None => return Err(GlrMaskError::GrammarParse("unterminated string".into())),
            }
        }
    }

    fn lex_regex(&mut self) -> Result<String, GlrMaskError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(b'/') => return Ok(s),
                Some(b'\\') => {
                    s.push('\\');
                    if let Some(b) = self.advance() {
                        s.push(b as char);
                    }
                }
                Some(b) => s.push(b as char),
                None => return Err(GlrMaskError::GrammarParse("unterminated regex".into())),
            }
        }
    }

    fn lex_ident(&mut self, first: u8) -> String {
        let mut s = String::new();
        s.push(first as char);
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                s.push(b as char);
                self.pos += 1;
            } else {
                break;
            }
        }
        s
    }

    fn lex_number(&mut self, first: u8) -> usize {
        let mut n = (first - b'0') as usize;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                n = n * 10 + (b - b'0') as usize;
                self.pos += 1;
            } else {
                break;
            }
        }
        n
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_inline();
            match self.peek() {
                None => break,
                Some(b'/') => {
                    self.pos += 1;
                    if self.peek() == Some(b'/') {
                        self.skip_comment();
                    } else {
                        let rx = self.lex_regex()?;
                        tokens.push(Token::Regex(rx));
                    }
                }
                Some(b'#') => self.skip_comment(),
                Some(b'%') => {
                    // Skip %ignore and other directives (rest of line).
                    self.pos += 1;
                    self.skip_comment();
                }
                Some(b'\n') => {
                    self.pos += 1;
                    tokens.push(Token::Newline);
                }
                Some(b'"') => {
                    self.pos += 1;
                    let s = self.lex_string(b'"')?;
                    tokens.push(Token::Literal(s));
                }
                Some(b'(') => {
                    self.pos += 1;
                    tokens.push(Token::LParen);
                }
                Some(b')') => {
                    self.pos += 1;
                    tokens.push(Token::RParen);
                }
                Some(b'[') => {
                    self.pos += 1;
                    tokens.push(Token::LBracket);
                }
                Some(b']') => {
                    self.pos += 1;
                    tokens.push(Token::RBracket);
                }
                Some(b'|') => {
                    self.pos += 1;
                    tokens.push(Token::Pipe);
                }
                Some(b'*') => {
                    self.pos += 1;
                    tokens.push(Token::Star);
                }
                Some(b'+') => {
                    self.pos += 1;
                    tokens.push(Token::Plus);
                }
                Some(b'?') => {
                    self.pos += 1;
                    tokens.push(Token::Question);
                }
                Some(b'.') => {
                    self.pos += 1;
                    tokens.push(Token::Dot);
                }
                Some(b'~') => {
                    self.pos += 1;
                    tokens.push(Token::Tilde);
                }
                Some(b',') => {
                    self.pos += 1;
                    tokens.push(Token::Comma);
                }
                Some(b'-') => {
                    self.pos += 1;
                    if self.peek() == Some(b'>') {
                        self.pos += 1;
                        tokens.push(Token::Arrow);
                    } else {
                        return Err(GlrMaskError::GrammarParse("unexpected '-'".into()));
                    }
                }
                Some(b':') => {
                    self.pos += 1;
                    tokens.push(Token::Colon);
                }
                Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                    self.pos += 1;
                    let ident = self.lex_ident(b);
                    if ident
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
                    {
                        tokens.push(Token::Terminal(ident));
                    } else {
                        tokens.push(Token::Ident(ident));
                    }
                }
                Some(b) if b.is_ascii_digit() => {
                    self.pos += 1;
                    let n = self.lex_number(b);
                    tokens.push(Token::Number(n));
                }
                Some(b) => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unexpected character '{}' at position {}",
                        b as char, self.pos
                    )));
                }
            }
        }
        Ok(tokens)
    }
}

// ---------------------------------------------------------------------------
// Tilde repetition desugaring
// ---------------------------------------------------------------------------

/// Desugar `expr~min` (exact) or `expr~min..max` (bounded) into existing
/// GrammarExpr types.
///
/// - `expr~N`       → `Seq([expr; N])`
/// - `expr~min..max` → `Seq([expr; min] ++ nested_optional(expr, max-min))`
///
/// When `max` is `None`, it means exact repetition with count `min`.
fn desugar_tilde(atom: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
    let max = max.unwrap_or(min);
    assert!(max >= min, "tilde max must be >= min");

    let mut parts: Vec<GrammarExpr> = Vec::with_capacity(max);
    // Mandatory copies.
    for _ in 0..min {
        parts.push(atom.clone());
    }
    // Optional tail: nested right-to-left.
    let extra = max - min;
    if extra > 0 {
        let mut tail = GrammarExpr::Optional(Box::new(atom.clone()));
        for _ in 1..extra {
            tail = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                atom.clone(),
                tail,
            ])));
        }
        parts.push(tail);
    }

    match parts.len() {
        0 => GrammarExpr::Sequence(vec![]),
        1 => parts.into_iter().next().unwrap(),
        _ => GrammarExpr::Sequence(parts),
    }
}

// ---------------------------------------------------------------------------
// Parser: tokens → NamedGrammar
// ---------------------------------------------------------------------------

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos)?.clone();
        self.pos += 1;
        Some(tok)
    }

    fn expect_token(&mut self, expected: &Token) -> Result<(), GlrMaskError> {
        match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            Some(tok) => Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, got {:?}",
                expected, tok
            ))),
            None => Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, got end of input",
                expected
            ))),
        }
    }

    fn skip_newlines(&mut self) {
        while self.peek() == Some(&Token::Newline) {
            self.pos += 1;
        }
    }

    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();

        self.skip_newlines();
        while self.pos < self.tokens.len() {
            let name = match self.advance() {
                Some(Token::Ident(s)) => s,
                Some(Token::Terminal(s)) => s,
                Some(other) => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "expected rule name, got {:?}",
                        other
                    )));
                }
                None => break,
            };

            self.expect_token(&Token::Colon)?;

            let expr = self.parse_alternatives()?;

            // Skip optional alias (-> name).
            if self.peek() == Some(&Token::Arrow) {
                self.pos += 1;
                // Skip the alias name.
                match self.advance() {
                    Some(Token::Ident(_)) | Some(Token::Terminal(_)) => {}
                    _ => {}
                }
            }

            rules.push((name, expr));
            self.skip_newlines();
        }

        if rules.is_empty() {
            return Err(GlrMaskError::GrammarParse("empty grammar".into()));
        }

        // Prefer a rule named "start"; fall back to the first rule.
        let start = if rules.iter().any(|(name, _)| name == "start") {
            "start".to_string()
        } else {
            rules[0].0.clone()
        };
        Ok(NamedGrammar { rules, start })
    }

    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let first = self.parse_sequence()?;
        let mut alts = vec![first];

        loop {
            // Direct pipe continuation.
            if self.peek() == Some(&Token::Pipe) {
                self.pos += 1;
                alts.push(self.parse_sequence()?);
                continue;
            }
            // Multi-line continuation: newline(s) then pipe.
            let saved = self.pos;
            let mut saw_newline = false;
            while self.peek() == Some(&Token::Newline) {
                self.pos += 1;
                saw_newline = true;
            }
            if saw_newline && self.peek() == Some(&Token::Pipe) {
                self.pos += 1;
                alts.push(self.parse_sequence()?);
                continue;
            }
            // Not a continuation — restore position.
            self.pos = saved;
            break;
        }

        if alts.len() == 1 {
            Ok(alts.into_iter().next().unwrap())
        } else {
            Ok(GrammarExpr::Choice(alts))
        }
    }

    fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let mut parts = Vec::new();

        while self.is_unit_start() {
            parts.push(self.parse_unit()?);
        }

        if parts.is_empty() {
            Ok(GrammarExpr::Sequence(vec![]))
        } else if parts.len() == 1 {
            Ok(parts.into_iter().next().unwrap())
        } else {
            Ok(GrammarExpr::Sequence(parts))
        }
    }

    fn is_unit_start(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Ident(_))
                | Some(Token::Terminal(_))
                | Some(Token::Literal(_))
                | Some(Token::Regex(_))
                | Some(Token::LParen)
                | Some(Token::LBracket)
                | Some(Token::Dot)
        )
    }

    fn parse_unit(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let atom = self.parse_atom()?;

        match self.peek() {
            Some(Token::Star) => {
                self.pos += 1;
                Ok(GrammarExpr::Repeat(Box::new(atom)))
            }
            Some(Token::Plus) => {
                self.pos += 1;
                Ok(GrammarExpr::RepeatOne(Box::new(atom)))
            }
            Some(Token::Question) => {
                self.pos += 1;
                Ok(GrammarExpr::Optional(Box::new(atom)))
            }
            Some(Token::Tilde) => {
                // Bounded repetition: expr~N or expr~N..M
                self.pos += 1;
                let min = match self.advance() {
                    Some(Token::Number(n)) => n,
                    _ => return Err(GlrMaskError::GrammarParse("expected number after ~".into())),
                };
                // Check for ..M
                let max = if self.peek() == Some(&Token::Dot) {
                    let saved = self.pos;
                    self.pos += 1;
                    if self.peek() == Some(&Token::Dot) {
                        self.pos += 1;
                        match self.advance() {
                            Some(Token::Number(m)) => Some(m),
                            _ => {
                                return Err(GlrMaskError::GrammarParse(
                                    "expected number after ..".into(),
                                ))
                            }
                        }
                    } else {
                        // Single dot — not a range; restore.
                        self.pos = saved;
                        None
                    }
                } else {
                    None
                };
                Ok(desugar_tilde(atom, min, max))
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(s)) => Ok(GrammarExpr::Literal(s.into_bytes())),
            Some(Token::Regex(rx)) => {
                // Use RawRegex to preserve the pattern as-is (avoids double-wrapping brackets).
                Ok(GrammarExpr::RawRegex(rx))
            }
            Some(Token::Dot) => Ok(GrammarExpr::AnyByte),
            Some(Token::LParen) => {
                let expr = self.parse_alternatives()?;
                self.expect_token(&Token::RParen)?;
                Ok(expr)
            }
            Some(Token::LBracket) => {
                let expr = self.parse_alternatives()?;
                self.expect_token(&Token::RBracket)?;
                Ok(GrammarExpr::Optional(Box::new(expr)))
            }
            other => Err(GlrMaskError::GrammarParse(format!(
                "expected atom, got {:?}",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a Lark grammar string into a `GrammarDef`.
pub fn parse_lark(input: &str) -> Result<GrammarDef, GlrMaskError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let named = parser.parse_grammar()?;
    lower(&named)
}

#[allow(dead_code)]
    /// Parse a Lark grammar string into a `NamedGrammar`.
pub fn parse_lark_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_grammar()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let g = parse_lark(r#"start: "a" "b""#).unwrap();
        assert_eq!(g.num_terminals(), 2);
    }

    #[test]
    fn test_parse_choice() {
        let g = parse_lark(r#"start: "a" | "b""#).unwrap();
        let start_rules: Vec<_> = g.rules.iter().filter(|r| r.lhs == g.start).collect();
        assert_eq!(start_rules.len(), 2);
    }

    #[test]
    fn test_parse_multi_rule() {
        let g = parse_lark(
            r#"
            start: item "."
            item: "a" | "b"
            "#,
        )
        .unwrap();
        assert!(g.num_nonterminals() >= 2);
    }

    #[test]
    fn test_parse_terminal_rule() {
        let g = parse_lark(
            r#"
            start: WORD
            WORD: /[a-z]+/
            "#,
        )
        .unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_repetition() {
        let g = parse_lark(r#"start: "a"+ "b"*"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_comments() {
        let g = parse_lark(
            r#"
            // This is a comment
            start: "a"  // inline
            "#,
        )
        .unwrap();
        assert_eq!(g.num_terminals(), 1);
    }

    #[test]
    fn test_parse_optional_bracket() {
        let g = parse_lark(r#"start: "a" ["b"]"#).unwrap();
        assert!(!g.rules.is_empty());
    }
}
