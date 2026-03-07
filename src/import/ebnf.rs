























#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This file is the direct glrmask analogue of sep1's `interface/ebnf.rs` parser.

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, lower};





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
    Separator, 
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
        unimplemented!()
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
        unimplemented!()
    }

    fn lex_ident(&mut self, first: u8) -> String {
        unimplemented!()
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {
        unimplemented!()
    }
}

fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {
    unimplemented!()
}





struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        unimplemented!()
    }

    fn peek(&self) -> Option<&Token> {
        unimplemented!()
    }

    fn advance(&mut self) -> Option<Token> {
        unimplemented!()
    }

    fn expect(&mut self, expected: &Token) -> Result<(), GlrMaskError> {
        unimplemented!()
    }

    fn skip_newlines(&mut self) {
        unimplemented!()
    }

    
    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        unimplemented!()
    }

    
    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    fn is_unit_start(&self) -> bool {
        unimplemented!()
    }

    
    fn parse_unit(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }

    
    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        unimplemented!()
    }
}






pub fn parse_ebnf(input: &str) -> Result<GrammarDef, GlrMaskError> {
    unimplemented!()
}

#[allow(dead_code)]
    
pub fn parse_ebnf_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    unimplemented!()
}





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
        assert_eq!(g.num_terminals(), 1); 
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
