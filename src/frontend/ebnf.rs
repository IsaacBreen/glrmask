//! EBNF grammar parser.
//!
//! Parses Extended Backus-Naur Form grammars into the internal `GrammarDef` IR.
//!
//! # Supported syntax
//!
//! ```text
//! # Comments start with #
//! rule_name ::= expression
//! rule_name : expression         # colon also accepted
//!
//! # Expressions
//! a b c           # sequence (concatenation)
//! a | b | c       # choice (alternation)
//! (expr)          # grouping
//! expr?           # optional (zero or one)
//! expr*           # Kleene star (zero or more)
//! expr+           # one or more
//! "text"          # literal string (double quotes)
//! 'text'          # literal string (single quotes)
//! [a-z]           # character class
//! [^a-z]          # negated character class
//! rule_name       # rule reference
//! ```

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::frontend::grammar_expr::{GrammarExpr, NamedGrammar, lower};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Literal(String),
    CharClass { def: String, negate: bool },
    LParen,
    RParen,
    Pipe,
    Star,
    Plus,
    Question,
    Separator, // ::= or :
    Newline,
    Dot,
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
                    Some(b'\'') => s.push('\''),
                    Some(b'x') => {
                        let hi = self.advance().ok_or_else(|| {
                            GlrMaskError::GrammarParse("unexpected end in hex escape".into())
                        })?;
                        let lo = self.advance().ok_or_else(|| {
                            GlrMaskError::GrammarParse("unexpected end in hex escape".into())
                        })?;
                        let val = hex_digit(hi)? * 16 + hex_digit(lo)?;
                        s.push(val as char);
                    }
                    Some(c) => {
                        s.push('\\');
                        s.push(c as char);
                    }
                    None => {
                        return Err(GlrMaskError::GrammarParse(
                            "unexpected end of input in string escape".into(),
                        ));
                    }
                },
                Some(b) => s.push(b as char),
                None => {
                    return Err(GlrMaskError::GrammarParse(
                        "unterminated string literal".into(),
                    ));
                }
            }
        }
    }

    fn lex_char_class(&mut self) -> Result<(String, bool), GlrMaskError> {
        let negate = if self.peek() == Some(b'^') {
            self.pos += 1;
            true
        } else {
            false
        };
        let mut def = String::new();
        let mut first = true;
        loop {
            match self.advance() {
                Some(b']') if !first => return Ok((def, negate)),
                Some(b'\\') => {
                    def.push('\\');
                    if let Some(b) = self.advance() {
                        def.push(b as char);
                    }
                }
                Some(b) => {
                    def.push(b as char);
                }
                None => {
                    return Err(GlrMaskError::GrammarParse(
                        "unterminated character class".into(),
                    ));
                }
            }
            first = false;
        }
    }

    fn lex_ident(&mut self, first: u8) -> String {
        let mut s = String::new();
        s.push(first as char);
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                s.push(b as char);
                self.pos += 1;
            } else {
                break;
            }
        }
        s
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_inline();
            match self.peek() {
                None => break,
                Some(b'#') => self.skip_comment(),
                Some(b'\n') => {
                    self.pos += 1;
                    tokens.push(Token::Newline);
                }
                Some(b'"') => {
                    self.pos += 1;
                    let s = self.lex_string(b'"')?;
                    tokens.push(Token::Literal(s));
                }
                Some(b'\'') => {
                    self.pos += 1;
                    let s = self.lex_string(b'\'')?;
                    tokens.push(Token::Literal(s));
                }
                Some(b'[') => {
                    self.pos += 1;
                    let (def, neg) = self.lex_char_class()?;
                    tokens.push(Token::CharClass { def, negate: neg });
                }
                Some(b'(') => {
                    self.pos += 1;
                    tokens.push(Token::LParen);
                }
                Some(b')') => {
                    self.pos += 1;
                    tokens.push(Token::RParen);
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
                Some(b':') => {
                    self.pos += 1;
                    if self.peek() == Some(b':') {
                        self.pos += 1;
                        if self.peek() == Some(b'=') {
                            self.pos += 1;
                        }
                    }
                    tokens.push(Token::Separator);
                }
                Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                    self.pos += 1;
                    let ident = self.lex_ident(b);
                    tokens.push(Token::Ident(ident));
                }
                Some(b) => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unexpected character '{}' (0x{:02x}) at position {}",
                        b as char, b, self.pos
                    )));
                }
            }
        }
        Ok(tokens)
    }
}

fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(GlrMaskError::GrammarParse(format!(
            "invalid hex digit '{}'",
            b as char
        ))),
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

    fn expect(&mut self, expected: &Token) -> Result<(), GlrMaskError> {
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

    /// Parse the full grammar.
    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();

        self.skip_newlines();
        while self.pos < self.tokens.len() {
            let name = match self.advance() {
                Some(Token::Ident(s)) => s,
                Some(other) => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "expected rule name, got {:?}",
                        other
                    )));
                }
                None => break,
            };

            self.expect(&Token::Separator)?;

            let expr = self.parse_alternatives()?;
            rules.push((name, expr));

            self.skip_newlines();
        }

        if rules.is_empty() {
            return Err(GlrMaskError::GrammarParse("empty grammar".into()));
        }

        let start = rules[0].0.clone();
        Ok(NamedGrammar { rules, start })
    }

    /// `alternatives = sequence ("|" sequence)*`
    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let first = self.parse_sequence()?;
        let mut alts = vec![first];

        while self.peek() == Some(&Token::Pipe) {
            self.pos += 1;
            alts.push(self.parse_sequence()?);
        }

        if alts.len() == 1 {
            Ok(alts.into_iter().next().unwrap())
        } else {
            Ok(GrammarExpr::Choice(alts))
        }
    }

    /// `sequence = unit+`
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
                | Some(Token::Literal(_))
                | Some(Token::CharClass { .. })
                | Some(Token::LParen)
                | Some(Token::Dot)
        )
    }

    /// `unit = atom quantifier?`
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
            _ => Ok(atom),
        }
    }

    /// `atom = IDENT | LITERAL | CHARCLASS | "(" alternatives ")" | "."`
    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(s)) => Ok(GrammarExpr::Literal(s.into_bytes())),
            Some(Token::CharClass { def, negate }) => Ok(GrammarExpr::CharClass { def, negate }),
            Some(Token::Dot) => Ok(GrammarExpr::AnyByte),
            Some(Token::LParen) => {
                let expr = self.parse_alternatives()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
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

/// Parse an EBNF grammar string into a `GrammarDef`.
pub fn parse_ebnf(input: &str) -> Result<GrammarDef, GlrMaskError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let named = parser.parse_grammar()?;
    lower(&named)
}

#[allow(dead_code)]
    /// Parse an EBNF grammar string into a `NamedGrammar` (intermediate form).
pub fn parse_ebnf_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
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
        let g = parse_ebnf("start ::= \"a\" \"b\"").unwrap();
        assert_eq!(g.num_terminals(), 2);
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_colon_separator() {
        let g = parse_ebnf("start : \"a\"").unwrap();
        assert_eq!(g.num_terminals(), 1);
    }

    #[test]
    fn test_parse_choice() {
        let g = parse_ebnf("start ::= \"a\" | \"b\"").unwrap();
        let start_rules: Vec<_> = g.rules.iter().filter(|r| r.lhs == g.start).collect();
        assert_eq!(start_rules.len(), 2);
    }

    #[test]
    fn test_parse_repetition() {
        let g = parse_ebnf("start ::= \"a\"+ \"b\"*").unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_multi_rule() {
        let g = parse_ebnf(
            r#"
            start ::= item "."
            item ::= "a" | "b"
            "#,
        )
        .unwrap();
        assert!(g.num_nonterminals() >= 2);
    }

    #[test]
    fn test_parse_char_class() {
        let g = parse_ebnf("start ::= [a-z]+").unwrap();
        assert_eq!(g.num_terminals(), 1);
    }

    #[test]
    fn test_parse_negated_char_class() {
        let g = parse_ebnf("start ::= [^0-9]+").unwrap();
        assert_eq!(g.num_terminals(), 1);
    }

    #[test]
    fn test_parse_optional() {
        let g = parse_ebnf("start ::= \"a\" \"b\"?").unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_grouping() {
        let g = parse_ebnf("start ::= (\"a\" | \"b\") \"c\"").unwrap();
        assert!(g.num_terminals() >= 2);
    }

    #[test]
    fn test_parse_comments() {
        let g = parse_ebnf(
            r#"
            # This is a comment
            start ::= "a"  # inline comment
            "#,
        )
        .unwrap();
        assert_eq!(g.num_terminals(), 1);
    }

    #[test]
    fn test_parse_escape_sequences() {
        let g = parse_ebnf(r#"start ::= "\n" "\t""#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_parse_empty_grammar_error() {
        assert!(parse_ebnf("# just a comment").is_err());
    }

    #[test]
    fn test_parse_dot_wildcard() {
        let g = parse_ebnf("start ::= . .").unwrap();
        assert_eq!(g.num_terminals(), 1); // both dots share the _any terminal
    }

    #[test]
    fn test_roundtrip_simple_ab() {
        let g = parse_ebnf("start ::= \"a\" \"b\"").unwrap();
        assert_eq!(g.start, 0);
        assert_eq!(g.num_terminals(), 2);
        let start_rules: Vec<_> = g.rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(start_rules.len(), 1);
        assert_eq!(start_rules[0].rhs.len(), 2);
    }
}
