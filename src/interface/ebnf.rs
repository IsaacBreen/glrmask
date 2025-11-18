use crate::interface::GrammarExpr::CharClass;
use crate::interface::{choice, literal, optional, r#ref, repeat, sequence, GrammarExpr};
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
pub(super) struct ParseError {
    message: String,
    span: Span,
}

#[derive(Debug, Clone, PartialEq)]
enum EbnfTokenKind {
    Ident(String),
    Literal(String),
    CharClass(String),
    Op(String),
}
#[derive(Debug, Clone, PartialEq)]
struct EbnfToken {
    kind: EbnfTokenKind,
    span: Span,
}

impl From<ParseError> for String {
    fn from(e: ParseError) -> Self {
        format!(
            "Parse error at byte range {}-{}: {}",
            e.span.start, e.span.end, e.message
        )
    }
}

fn get_token_regex() -> &'static Regex {
    static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
    TOKEN_REGEX.get_or_init(|| {
        Regex::new(
            r#"(?x)
        (?P<ident>[a-zA-Z_][a-zA-Z0-9_]*) |
        (?P<literal>"([^"\\]|\\.)*"|'([^'\\]|\\.)*') |
        (?P<charclass>\[([^\]\\()\s]|\\.)*\]) |
        (?P<op>::=|;|\?|\*|\+|\||\(|\)|\[|\]|\{|\}|!|\.|\#) |
        (?P<comment>//[^\r\n]*|/\*([^*]|\*[^/])*\*/) |
        (?P<ws>\s+) |
        (?P<error>.)
        "#,
        )
        .unwrap()
    })
}

fn tokenize(source: &str) -> Result<Vec<EbnfToken>, ParseError> {
    let mut tokens = Vec::new();
    for cap in get_token_regex().captures_iter(source) {
        if let Some(m) = cap.name("ident") {
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Ident(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("literal") {
            let s = m.as_str();
            let content = &s[1..s.len() - 1];
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
                            other => unescaped.push(other),
                        }
                    } else {
                        // This case should be prevented by the regex, but as a safeguard:
                        return Err(ParseError {
                            message: format!("Literal with dangling escape: {}", s),
                            span: Span { start: m.start(), end: m.end() },
                        });
                    }
                } else {
                    unescaped.push(c);
                }
            }
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Literal(unescaped),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("charclass") {
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::CharClass(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("op") {
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Op(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(e) = cap.name("error") {
            return Err(ParseError {
                message: format!("Unknown token: {}", e.as_str()),
                span: Span { start: e.start(), end: e.end() },
            });
        }
        // ws and comment are ignored
    }
    Ok(tokens)
}

#[derive(Debug)]
pub(super) struct EbnfParseResult {
    pub grammar_rules: Vec<(String, GrammarExpr)>,
    pub ignore_symbol_name: Option<String>,
}

pub(super) struct EbnfParser<'a> {
    source: &'a str,
    tokens: Peekable<IntoIter<EbnfToken>>,
}

impl<'a> EbnfParser<'a> {
    pub(super) fn new(source: &'a str) -> Result<Self, ParseError> {
        let tokens = tokenize(source)?;
        Ok(EbnfParser {
            source,
            tokens: tokens.into_iter().peekable(),
        })
    }

    fn parse_rule_body(&mut self) -> Result<GrammarExpr, ParseError> {
        self.expect_grammar_op("::=")?;
        let expr = self.parse_grammar_expression()?;
        self.expect_grammar_op(";")?;
        Ok(expr)
    }

    pub(super) fn parse(&mut self) -> Result<EbnfParseResult, ParseError> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
        let mut seen_names = HashSet::new();
        let mut ignore_symbol_name = None;

        while self.tokens.peek().is_some() {
            if self.peek_grammar_op("#") {
                let directive_span = self.tokens.peek().unwrap().span;
                self.consume_grammar_op("#")?;
                self.expect_grammar_op("!")?;
                self.expect_grammar_op("[")?;
                if ignore_symbol_name.is_some() {
                    return Err(ParseError {
                        message: "Duplicate ignore directive found".to_string(),
                        span: directive_span,
                    });
                }
                let (directive_name, directive_name_span) = self.expect_ident()?;
                if directive_name != "ignore" {
                    return Err(ParseError {
                        message: format!("Unknown directive: {}", directive_name),
                        span: directive_name_span,
                    });
                }
                self.expect_grammar_op("(")?;
                let (symbol_name, _) = self.expect_ident()?;
                self.expect_grammar_op(")")?;
                self.expect_grammar_op("]")?;
                ignore_symbol_name = Some(symbol_name);
            } else {
                let (rule_name, rule_name_span) = self.expect_ident()?;
                if seen_names.contains(&rule_name) {
                    return Err(ParseError {
                        message: format!("Duplicate rule name: {}", rule_name),
                        span: rule_name_span,
                    });
                }
                seen_names.insert(rule_name.clone());
                let rule_expr = self.parse_rule_body()?;
                rules.push((rule_name, rule_expr));
            }
        }

        Ok(EbnfParseResult {
            grammar_rules: rules,
            ignore_symbol_name,
        })
    }

    fn parse_grammar_expression(&mut self) -> Result<GrammarExpr, ParseError> {
        let mut choices = vec![self.parse_grammar_sequence()?];
        while self.peek_grammar_op("|") {
            self.consume_grammar_op("|")?;
            choices.push(self.parse_grammar_sequence()?);
        }
        if choices.len() == 1 {
            Ok(choices.remove(0))
        } else {
            Ok(choice(choices))
        }
    }

    fn parse_grammar_sequence(&mut self) -> Result<GrammarExpr, ParseError> {
        let mut terms = Vec::new();
        // A sequence can be empty, which is a valid choice in an expression (e.g., `A ::= B | ;`)
        while self.tokens.peek().is_some()
            && !self.peek_grammar_op(")")
            && !self.peek_grammar_op("]")
            && !self.peek_grammar_op("}")
            && !self.peek_grammar_op("|")
            && !self.peek_grammar_op(";")
        {
            terms.push(self.parse_grammar_term()?);
        }

        if terms.len() == 1 {
            Ok(terms.remove(0))
        } else {
            Ok(sequence(terms))
        }
    }

    fn parse_grammar_term(&mut self) -> Result<GrammarExpr, ParseError> {
        let factor = self.parse_grammar_factor()?;

        if self.peek_grammar_op("?") {
            self.consume_grammar_op("?")?;
            Ok(optional(factor))
        } else if self.peek_grammar_op("*") {
            self.consume_grammar_op("*")?;
            Ok(repeat(factor))
        } else if self.peek_grammar_op("+") {
            self.consume_grammar_op("+")?;
            Ok(sequence(vec![factor.clone(), repeat(factor)]))
        } else {
            Ok(factor)
        }
    }

    fn parse_grammar_factor(&mut self) -> Result<GrammarExpr, ParseError> {
        if self.peek_grammar_op(".") {
            self.consume_grammar_op(".")?;
            Ok(GrammarExpr::AnyChar)
        } else if let Some(EbnfToken { kind: EbnfTokenKind::Ident(id), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(r#ref(&id))
        } else if let Some(EbnfToken { kind: EbnfTokenKind::Literal(lit), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(literal(lit.into_bytes()))
        } else if let Some(EbnfToken { kind: EbnfTokenKind::CharClass(cc), .. }) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(CharClass(cc))
        } else if self.peek_grammar_op("(") {
            self.consume_grammar_op("(")?;
            let expr = self.parse_grammar_expression()?;
            self.expect_grammar_op(")")?;
            Ok(expr)
        } else if self.peek_grammar_op("[") {
            self.consume_grammar_op("[")?;
            let expr = self.parse_grammar_expression()?;
            self.expect_grammar_op("]")?;
            Ok(optional(expr))
        } else if self.peek_grammar_op("{") {
            self.consume_grammar_op("{")?;
            let expr = self.parse_grammar_expression()?;
            self.expect_grammar_op("}")?;
            Ok(repeat(expr))
        } else {
            let (message, span) = if let Some(token) = self.tokens.peek() {
                (
                    format!(
                        "Expected identifier, literal, group, or '.', found {:?}",
                        &token.kind
                    ),
                    token.span,
                )
            } else {
                (
                    "Expected identifier, literal, group, or '.', found end of input".to_string(),
                    self.eof_span(),
                )
            };
            Err(ParseError { message, span })
        }
    }

    // --- Parser Helpers ---

    fn eof_span(&self) -> Span {
        let end = self.source.len();
        Span { start: end, end }
    }

    fn peek_grammar_op(&mut self, op: &str) -> bool {
        matches!(self.tokens.peek(), Some(EbnfToken { kind: EbnfTokenKind::Op(s), .. }) if s == op)
    }

    fn consume_grammar_op(&mut self, op: &str) -> Result<(), ParseError> {
        if self.peek_grammar_op(op) {
            self.tokens.next();
            Ok(())
        } else {
            let (message, span) = if let Some(token) = self.tokens.peek() {
                (
                    format!("Expected op '{}', found {:?}", op, &token.kind),
                    token.span,
                )
            } else {
                (
                    format!("Expected op '{}', found end of input", op),
                    self.eof_span(),
                )
            };
            Err(ParseError { message, span })
        }
    }

    fn expect_grammar_op(&mut self, op: &str) -> Result<(), ParseError> {
        match self.tokens.next() {
            Some(EbnfToken { kind: EbnfTokenKind::Op(s), .. }) if s == op => Ok(()),
            Some(other) => Err(ParseError {
                message: format!("Expected op '{}', found {:?}", op, other.kind),
                span: other.span,
            }),
            None => Err(ParseError {
                message: format!("Expected op '{}', found end of input", op),
                span: self.eof_span(),
            }),
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.tokens.next() {
            Some(EbnfToken { kind: EbnfTokenKind::Ident(id), span }) => Ok((id, span)),
            Some(other) => Err(ParseError {
                message: format!("Expected identifier, found {:?}", other.kind),
                span: other.span,
            }),
            None => Err(ParseError {
                message: "Expected identifier, found end of input".to_string(),
                span: self.eof_span(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::{choice, literal, optional, r#ref, repeat, sequence, GrammarDefinition};

    #[test]
    fn test_ebnf_parser_simple() {
        let ebnf = r#"
            s ::= a b;
            a ::= 'a' | ;
            b ::= c*;
            c ::= 'c'?;
        "#;
        let mut parser = EbnfParser::new(ebnf).unwrap();
        let rules = parser.parse().unwrap().grammar_rules;

        let expected_rules = vec![
            ("s".to_string(), sequence(vec![r#ref("a"), r#ref("b")])),
            (
                "a".to_string(),
                choice(vec![literal(b"a".to_vec()), sequence(vec![])]),
            ),
            ("b".to_string(), repeat(r#ref("c"))),
            ("c".to_string(), optional(literal(b"c".to_vec()))),
        ];

        assert_eq!(rules, expected_rules);
    }

    #[should_panic]
    #[test]
    fn test_ebnf_parser_error_with_span() {
        let ebnf = r#"
            s ::= a b;
            a ::= 'a' | ;
            b ::= c*;
            c ::= 'c'??;
        "#;
        let mut parser = EbnfParser::new(ebnf).unwrap();
        let err = parser.parse().unwrap_err();
        assert!(err
            .message
            .contains("Expected identifier, literal, group, or '.'"));
        assert_eq!(err.span.start, 82);
    }
}

