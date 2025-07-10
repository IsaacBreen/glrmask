use crate::interface::{choice, choice_fast, literal, optional, r#ref, repeat, sequence, GrammarExpr};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::iter::Peekable;
use std::sync::OnceLock;
use std::vec::IntoIter;
use crate::finite_automata::{Expr, QuantifierType};

#[derive(Debug, Clone, PartialEq)]
enum EbnfToken {
    Ident(String),
    Literal(String),
    Op(String),
}

fn get_token_regex() -> &'static Regex {
    static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
    TOKEN_REGEX.get_or_init(|| {
        Regex::new(
            r#"(?x)
        (?P<ident>[a-zA-Z_][a-zA-Z0-9_]*) |
        (?P<literal>"([^"\\]|\\.)*"|'([^'\\]|\\.)*') |
        (?P<op>::=|;|\?|\*|\+|\||\(|\)|\[|\]|\{|\}|!) |
        (?P<comment>\(\*([^*]|[\r\n]|(\*+([^*)]|[\r\n])))*\*+\)) |
        (?P<ws>\s+) |
        (?P<error>.)
        "#,
        )
        .unwrap()
    })
}

fn tokenize(source: &str) -> Result<Vec<EbnfToken>, String> {
    let mut tokens = Vec::new();
    for cap in get_token_regex().captures_iter(source) {
        if let Some(ident) = cap.name("ident") {
            tokens.push(EbnfToken::Ident(ident.as_str().to_string()));
        } else if let Some(lit) = cap.name("literal") {
            let s = lit.as_str();
            let content = &s[1..s.len() - 1];
            let mut unescaped = String::with_capacity(content.len());
            let mut chars = content.chars();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    if let Some(next_c) = chars.next() {
                        unescaped.push(next_c);
                    } else {
                        // This case should be prevented by the regex, but as a safeguard:
                        return Err(format!("Literal with dangling escape: {}", s));
                    }
                } else {
                    unescaped.push(c);
                }
            }
            tokens.push(EbnfToken::Literal(unescaped));
        } else if let Some(op) = cap.name("op") {
            tokens.push(EbnfToken::Op(op.as_str().to_string()));
        } else if let Some(e) = cap.name("error") {
            return Err(format!("Unknown token: {}", e.as_str()));
        }
        // ws and comment are ignored
    }
    Ok(tokens)
}

pub(super) struct EbnfParseResult {
    pub grammar_rules: Vec<(String, GrammarExpr)>,
    pub ignore_symbol_name: Option<String>,
}

pub(super) struct EbnfParser {
    tokens: Peekable<IntoIter<EbnfToken>>,
}

impl EbnfParser {
    pub(super) fn new(source: &str) -> Result<Self, String> {
        let tokens = tokenize(source)?;
        Ok(EbnfParser {
            tokens: tokens.into_iter().peekable(),
        })
    }

    fn parse_rule(&mut self) -> Result<(String, GrammarExpr), String> {
        let name = self.expect_ident()?;
        self.expect_grammar_op("::=")?;
        let expr = self.parse_grammar_expression()?;
        self.expect_grammar_op(";")?;
        Ok((name, expr))
    }

    pub(super) fn parse(&mut self) -> Result<EbnfParseResult, String> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
        let mut seen_names = HashSet::new();
        let mut ignore_symbol_name = None;

        while self.tokens.peek().is_some() {
            if self.peek_grammar_op("!") {
                if ignore_symbol_name.is_some() {
                    return Err("Duplicate ignore directive found".to_string())
                }
                self.consume_grammar_op("!")?;
                let directive_name = self.expect_ident()?;
                if directive_name != "ignore" {
                    return Err(format!("Unknown directive: {}", directive_name))
                }
                let symbol_name = self.expect_ident()?;
                self.expect_grammar_op(";")?;
                ignore_symbol_name = Some(symbol_name);
            } else {
                let (rule_name, rule_expr) = self.parse_rule()?;
                if seen_names.contains(&rule_name) {
                    return Err(format!("Duplicate rule name: {}", rule_name));
                }
                seen_names.insert(rule_name.clone());
                rules.push((rule_name, rule_expr));
            }
        }

        Ok(EbnfParseResult {
            grammar_rules: rules,
            ignore_symbol_name,
        })
    }

    fn parse_grammar_expression(&mut self) -> Result<GrammarExpr, String> {
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

    fn parse_grammar_sequence(&mut self) -> Result<GrammarExpr, String> {
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

    fn parse_grammar_term(&mut self) -> Result<GrammarExpr, String> {
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

    fn parse_grammar_factor(&mut self) -> Result<GrammarExpr, String> {
        if let Some(EbnfToken::Ident(id)) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(r#ref(&id))
        } else if let Some(EbnfToken::Literal(lit)) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(literal(lit.into_bytes()))
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
            Err(format!(
                "Expected identifier, literal, or group, found {:?}",
                self.tokens.peek()
            ))
        }
    }

    // --- Parser Helpers ---

    fn peek_grammar_op(&mut self, op: &str) -> bool {
        matches!(self.tokens.peek(), Some(EbnfToken::Op(s)) if s == op)
    }

    fn consume_grammar_op(&mut self, op: &str) -> Result<(), String> {
        if self.peek_grammar_op(op) {
            self.tokens.next();
            Ok(())
        } else {
            Err(format!("Expected op '{}', found {:?}", op, self.tokens.peek()))
        }
    }

    fn expect_grammar_op(&mut self, op: &str) -> Result<(), String> {
        match self.tokens.next() {
            Some(EbnfToken::Op(s)) if s == op => Ok(()),
            other => Err(format!("Expected op '{}', found {:?}", op, other)),
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.tokens.next() {
            Some(EbnfToken::Ident(id)) => Ok(id),
            other => Err(format!("Expected identifier, found {:?}", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::{choice, literal, optional, r#ref, repeat, sequence, GrammarDefinition, GrammarExpr};

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

    #[test]
    fn test_from_ebnf_integration() {
        let ebnf = "start ::= ('a' 'b')+ ;";
        let grammar_def = GrammarDefinition::from_ebnf(ebnf).unwrap();

        // Check that it produced a reasonable grammar definition
        assert_eq!(grammar_def.productions.len(), 4); // start' -> start, start -> ..., new_nt -> ..., new_nt ->
        assert!(grammar_def
            .terminal_name_to_group_id
            .contains_left("\"a\""));
        assert!(grammar_def
            .terminal_name_to_group_id
            .contains_left("\"b\""));
    }
}
