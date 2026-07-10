use super::{choice_or_single, sequence_or_single};
use crate::GlrMaskError;
use crate::grammar::flat::GrammarDef;
use crate::grammar::factoring::factor_named_grammar;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, Quantifier, lower};

/// All-uppercase (plus underscores and digits) rule names are terminals.
fn is_terminal_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

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
                        let hi = self.advance().ok_or_else(|| {
                            GlrMaskError::GrammarParse("incomplete hex escape".into())
                        })?;
                        let lo = self.advance().ok_or_else(|| {
                            GlrMaskError::GrammarParse("incomplete hex escape".into())
                        })?;
                        let value = (hex_digit(hi)? << 4) | hex_digit(lo)?;
                        s.push(value as char);
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
        let mut negate = false;
        if self.peek() == Some(b'^') {
            negate = true;
            self.pos += 1;
        }

        let mut def = String::new();
        while let Some(byte) = self.advance() {
            if byte == b']' {
                return Ok((def, negate));
            }
            if byte == b'\\' {
                let escaped = self.advance().ok_or_else(|| {
                    GlrMaskError::GrammarParse("unterminated char class escape".into())
                })?;
                def.push('\\');
                def.push(escaped as char);
            } else {
                def.push(byte as char);
            }
        }
        Err(GlrMaskError::GrammarParse("unterminated char class".into()))
    }

    fn lex_ident(&mut self, first: u8) -> String {
        let mut ident = String::from(first as char);
        while let Some(byte) = self.peek() {
            if is_ebnf_ident_continue(byte) {
                ident.push(byte as char);
                self.pos += 1;
            } else {
                break;
            }
        }
        ident
    }

    fn lex_separator(&mut self) {
        if self.peek() == Some(b':') && self.input.get(self.pos + 1) == Some(&b'=') {
            self.pos += 2;
        } else if self.peek() == Some(b'=') {
            self.pos += 1;
        }
    }

    fn lex_literal_token(&mut self, quote: u8) -> Result<Token, GlrMaskError> {
        Ok(Token::Literal(self.lex_string(quote)?))
    }

    fn lex_char_class_token(&mut self) -> Result<Token, GlrMaskError> {
        let (def, negate) = self.lex_char_class()?;
        Ok(Token::CharClass { def, negate })
    }

    fn lex_ident_token(&mut self, first: u8) -> Token {
        Token::Ident(self.lex_ident(first))
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {
        let mut tokens = Vec::new();
        while let Some(byte) = self.advance() {
            match byte {
                b' ' | b'\t' | b'\r' => self.skip_whitespace_inline(),
                b'\n' => tokens.push(Token::Newline),
                b'#' => self.skip_comment(),
                b'(' => tokens.push(Token::LParen),
                b')' => tokens.push(Token::RParen),
                b'|' => tokens.push(Token::Pipe),
                b'*' => tokens.push(Token::Star),
                b'+' => tokens.push(Token::Plus),
                b'?' => tokens.push(Token::Question),
                b'.' => tokens.push(Token::Dot),
                b':' => {
                    self.lex_separator();
                    tokens.push(Token::Separator);
                }
                b'"' | b'\'' => tokens.push(self.lex_literal_token(byte)?),
                b'[' => tokens.push(self.lex_char_class_token()?),
                b if is_ebnf_ident_start(b) => tokens.push(self.lex_ident_token(b)),
                _ => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unexpected character '{}'",
                        byte as char
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
        b'a'..=b'f' => Ok(10 + (b - b'a')),
        b'A'..=b'F' => Ok(10 + (b - b'A')),
        _ => Err(GlrMaskError::GrammarParse(format!("invalid hex digit '{}'", b as char))),
    }
}

fn is_ebnf_ident_start(byte: u8) -> bool {
    (byte as char).is_ascii_alphabetic() || byte == b'_'
}

fn is_ebnf_ident_continue(byte: u8) -> bool {
    (byte as char).is_ascii_alphanumeric() || byte == b'_'
}

fn apply_postfix_operator(atom: GrammarExpr, token: Option<&Token>) -> GrammarExpr {
    match token {
        Some(Token::Question) => GrammarExpr::Quantified(Box::new(atom), Quantifier::Optional),
        Some(Token::Star) => GrammarExpr::Quantified(Box::new(atom), Quantifier::ZeroPlus),
        Some(Token::Plus) => GrammarExpr::Quantified(Box::new(atom), Quantifier::OnePlus),
        _ => atom,
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned()?;
        self.pos += 1;
        Some(token)
    }

    fn expect(&mut self, expected: &Token) -> Result<(), GlrMaskError> {
        let actual = self.advance().ok_or_else(|| {
            GlrMaskError::GrammarParse("unexpected end of input".into())
        })?;
        if &actual == expected {
            Ok(())
        } else {
            Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, found {:?}",
                expected, actual
            )))
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Token::Newline)) {
            self.pos += 1;
        }
    }

    fn parse_rule_name(&mut self) -> Result<String, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) => Ok(name),
            Some(token) => Err(GlrMaskError::GrammarParse(format!(
                "expected rule name, found {:?}",
                token
            ))),
            None => Err(GlrMaskError::GrammarParse(
                "expected rule name, found end of input".into(),
            )),
        }
    }

    fn parse_rule(&mut self) -> Result<NamedRule, GlrMaskError> {
        let name = self.parse_rule_name()?;
        self.expect(&Token::Separator)?;
        let expr = self.parse_alternatives()?;
        let is_terminal = is_terminal_name(&name);
        Ok(NamedRule {
            name,
            expr,
            is_terminal,
            is_internal: false,
        })
    }

    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        self.skip_newlines();
        let mut rules = Vec::new();
        while self.peek().is_some() {
            rules.push(self.parse_rule()?);
            self.skip_newlines();
        }

        let start = rules
            .first()
            .map(|r| r.name.clone())
            .ok_or_else(|| GlrMaskError::GrammarParse("empty grammar".into()))?;
        Ok(NamedGrammar {
            rules,
            start,
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        })
    }

    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let mut options = vec![self.parse_sequence()?];
        while matches!(self.peek(), Some(Token::Pipe)) {
            self.advance();
            options.push(self.parse_sequence()?);
        }
        Ok(choice_or_single(options))
    }

    fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let mut items = Vec::new();
        while self.is_unit_start() {
            items.push(self.parse_unit()?);
        }
        Ok(sequence_or_single(items))
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

    fn parse_unit(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let atom = self.parse_atom()?;
        let quantifier = self.peek().cloned();
        if matches!(
            quantifier,
            Some(Token::Question) | Some(Token::Star) | Some(Token::Plus)
        ) {
            self.advance();
        }
        Ok(apply_postfix_operator(atom, quantifier.as_ref()))
    }

    fn parse_group(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let expr = self.parse_alternatives()?;
        self.expect(&Token::RParen)?;
        Ok(expr)
    }

    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(literal)) => Ok(GrammarExpr::Literal(literal.into_bytes())),
            Some(Token::CharClass { def, negate }) => Ok(GrammarExpr::CharClass { def, negate, utf8: true }),
            Some(Token::Dot) => Ok(GrammarExpr::AnyByte),
            Some(Token::LParen) => self.parse_group(),
            Some(token) => Err(GlrMaskError::GrammarParse(format!(
                "unexpected token {:?}",
                token
            ))),
            None => Err(GlrMaskError::GrammarParse("unexpected end of input".into())),
        }
    }
}

pub fn parse_ebnf(input: &str) -> Result<GrammarDef, GlrMaskError> {
    let named = parse_ebnf_to_named(input)?;
    let factored = factor_named_grammar(named);
    lower(&factored)
}

pub fn parse_ebnf_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_grammar()
}
