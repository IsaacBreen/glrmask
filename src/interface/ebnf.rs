use crate::interface::{choice, literal, optional, r#ref, repeat, sequence, GrammarExpr};
use regex::Regex;
use std::collections::{BTreeMap, HashSet};
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
    pub resolved_terminals: BTreeMap<String, Expr>,
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
        self.expect_op("::=")?;
        let expr = self.parse_expression()?;
        self.expect_op(";")?;
        Ok((name, expr))
    }

    pub(super) fn parse(&mut self) -> Result<EbnfParseResult, String> {
        let mut non_terminal_rules = Vec::new();
        let mut all_terminal_defs = BTreeMap::new();
        let mut ignore_symbol_name = None;

        while self.tokens.peek().is_some() {
            if self.peek_op("!") {
                if ignore_symbol_name.is_some() {
                    return Err("Duplicate ignore directive found".to_string())
                }
                self.consume_op("!")?;
                let directive_name = self.expect_ident()?;
                if directive_name != "ignore" {
                    return Err(format!("Unknown directive: {}", directive_name))
                }
                let symbol_name = self.expect_ident()?;
                self.expect_op(";")?;
                ignore_symbol_name = Some(symbol_name);
            } else {
                let (name, expr) = self.parse_rule()?;
                if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    if all_terminal_defs.insert(name.clone(), expr).is_some() {
                        return Err(format!("Duplicate definition for terminal '{}'", name))
                    }
                } else {
                    non_terminal_rules.push((name, expr));
                }
            }
        }

        // Identify terminals referenced by non-terminal rules
        let mut referenced_terminals = HashSet::new();
        for (_, grammar_expr) in &non_terminal_rules {
            Self::collect_referenced_terminals(grammar_expr, &mut referenced_terminals);
        }

        // Resolve referenced terminals and their dependencies
        let mut resolved_terminals = BTreeMap::new();
        let mut visiting = HashSet::new(); // For cycle detection

        let terminal_defs_ref: BTreeMap<String, &GrammarExpr> =
            all_terminal_defs.iter().map(|(k, v)| (k.clone(), v)).collect();

        for terminal_name in referenced_terminals {
            Self::resolve_one_terminal(
                &terminal_name,
                &terminal_defs_ref,
                &mut resolved_terminals,
                &mut visiting,
            )?;
        }

        Ok(EbnfParseResult {
            grammar_rules: non_terminal_rules,
            resolved_terminals,
            ignore_symbol_name,
        })
    }

    fn parse_expression(&mut self) -> Result<GrammarExpr, String> {
        let mut choices = vec![self.parse_sequence()?];
        while self.peek_op("|") {
            self.consume_op("|")?;
            choices.push(self.parse_sequence()?);
        }
        if choices.len() == 1 {
            Ok(choices.remove(0))
        } else {
            Ok(choice(choices))
        }
    }

    fn parse_sequence(&mut self) -> Result<GrammarExpr, String> {
        let mut terms = Vec::new();
        // A sequence can be empty, which is a valid choice in an expression (e.g., `A ::= B | ;`)
        while self.tokens.peek().is_some()
            && !self.peek_op(")")
            && !self.peek_op("]")
            && !self.peek_op("}")
            && !self.peek_op("|")
            && !self.peek_op(";")
        {
            terms.push(self.parse_term()?);
        }

        if terms.len() == 1 {
            Ok(terms.remove(0))
        } else {
            Ok(sequence(terms))
        }
    }

    fn parse_term(&mut self) -> Result<GrammarExpr, String> {
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
        } else {
            Ok(factor)
        }
    }

    fn parse_factor(&mut self) -> Result<GrammarExpr, String> {
        if let Some(EbnfToken::Ident(id)) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(r#ref(&id))
        } else if let Some(EbnfToken::Literal(lit)) = self.tokens.peek().cloned() {
            self.tokens.next();
            Ok(literal(lit.into_bytes()))
        } else if self.peek_op("(") {
            self.consume_op("(")?;
            let expr = self.parse_expression()?;
            self.expect_op(")")?;
            Ok(expr)
        } else if self.peek_op("[") {
            self.consume_op("[")?;
            let expr = self.parse_expression()?;
            self.expect_op("]")?;
            Ok(optional(expr))
        } else if self.peek_op("{") {
            self.consume_op("{")?;
            let expr = self.parse_expression()?;
            self.expect_op("}")?;
            Ok(repeat(expr))
        } else {
            Err(format!(
                "Expected identifier, literal, or group, found {:?}",
                self.tokens.peek()
            ))
        }
    }

    // --- Parser Helpers ---

    fn peek_op(&mut self, op: &str) -> bool {
        matches!(self.tokens.peek(), Some(EbnfToken::Op(s)) if s == op)
    }

    fn consume_op(&mut self, op: &str) -> Result<(), String> {
        if self.peek_op(op) {
            self.tokens.next();
            Ok(())
        } else {
            Err(format!("Expected op '{}', found {:?}", op, self.tokens.peek()))
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<(), String> {
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

    fn collect_referenced_terminals(expr: &GrammarExpr, refs: &mut HashSet<String>) {
        match expr {
            GrammarExpr::Ref(name) => {
                if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    refs.insert(name.clone());
                }
            }
            GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                for e in exprs {
                    Self::collect_referenced_terminals(e, refs);
                }
            }
            GrammarExpr::Optional(e) | GrammarExpr::Repeat(e) => {
                Self::collect_referenced_terminals(e.as_ref(), refs);
            }
            GrammarExpr::Literal(_) | GrammarExpr::RegexExpr(_) => {}
        }
    }

    fn convert_terminal_expr(
        expr: &GrammarExpr,
        terminal_defs: &BTreeMap<String, &GrammarExpr>,
        resolved: &mut BTreeMap<String, Expr>,
        visiting: &mut HashSet<String>,
    ) -> Result<Expr, String> {
        match expr {
            GrammarExpr::Literal(bytes) => Ok(Expr::U8Seq(bytes.clone())),
            GrammarExpr::RegexExpr(_) => {
                Err("GrammarExpr::RegexExpr not allowed in terminal definitions.".to_string())
            }
            GrammarExpr::Ref(name) => {
                if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    Self::resolve_one_terminal(name, terminal_defs, resolved, visiting)
                } else {
                    Err(format!(
                        "Terminals cannot reference non-terminals ('{}')",
                        name
                    ))
                }
            }
            GrammarExpr::Sequence(exprs) => {
                let children = exprs
                    .iter()
                    .map(|e| Self::convert_terminal_expr(e, terminal_defs, resolved, visiting))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Seq(children))
            }
            GrammarExpr::Choice(exprs) => {
                let children = exprs
                    .iter()
                    .map(|e| Self::convert_terminal_expr(e, terminal_defs, resolved, visiting))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Choice(children))
            }
            GrammarExpr::Optional(expr) => {
                let child = Self::convert_terminal_expr(expr, terminal_defs, resolved, visiting)?;
                Ok(Expr::Quantifier(
                    Box::new(child),
                    QuantifierType::ZeroOrOne,
                ))
            }
            GrammarExpr::Repeat(expr) => {
                let child = Self::convert_terminal_expr(expr, terminal_defs, resolved, visiting)?;
                Ok(Expr::Quantifier(
                    Box::new(child),
                    QuantifierType::ZeroOrMore,
                ))
            }
        }
    }

    fn resolve_one_terminal<'a>(
        name: &'a str,
        terminal_defs: &'a BTreeMap<String, &GrammarExpr>,
        resolved: &'a mut BTreeMap<String, Expr>,
        visiting: &'a mut HashSet<String>,
    ) -> Result<Expr, String> {
        if let Some(expr) = resolved.get(name) {
            return Ok(expr.clone());
        }
        if visiting.contains(name) {
            return Err(format!(
                "Circular reference in terminal definitions involving '{}'",
                name
            ));
        }

        visiting.insert(name.to_string());

        let expr_def = terminal_defs
            .get(name)
            .ok_or_else(|| format!("Undefined terminal '{}' referenced.", name))?;
        let regex_expr = Self::convert_terminal_expr(expr_def, terminal_defs, resolved, visiting)?;

        visiting.remove(name);
        resolved.insert(name.to_string(), regex_expr.clone());
        Ok(regex_expr)
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
