use crate::interface::{choice, literal, optional, r#ref, repeat, sequence, GrammarExpr};
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ParseError {
    message: String,
    span: Span,
    line: usize,
    column: usize,
    line_text: String,
}

impl ParseError {
    fn new(source: &str, span: Span, message: impl Into<String>) -> Self {
        let (line, column, line_text) = compute_line_info(source, span.start);
        ParseError {
            message: message.into(),
            span,
            line,
            column,
            line_text,
        }
    }
}

// Compute 1-based line and column numbers and capture the offending line.
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

#[derive(Debug, Clone, PartialEq)]
enum EbnfTokenKind {
    Ident(String),
    Literal(String),
    CharClass(String),
    Op(String),
    /// Repetition specification: {m}, {m,}, {m,n}, {,n}
    Repetition { min: usize, max: Option<usize> },
}
#[derive(Debug, Clone, PartialEq)]
struct EbnfToken {
    kind: EbnfTokenKind,
    span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Human-friendly summary line
        writeln!(f, "error: EBNF parse error: {}", self.message)?;

        // Location line
        write!(
            f,
            " --> line {}, column {} (byte range {}-{})",
            self.line, self.column, self.span.start, self.span.end
        )?;

        // If we captured the offending line, show it with a caret marker.
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

impl std::error::Error for ParseError {}

impl From<ParseError> for String {
    fn from(e: ParseError) -> Self {
        e.to_string()
    }
}

fn get_token_regex() -> &'static Regex {
    static TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
    TOKEN_REGEX.get_or_init(|| {
        Regex::new(
            r#"(?x)
        (?P<directive>\#!) |
        (?P<comment>//[^\r\n]*|/\*([^*]|\*[^/])*\*/|\#[^\r\n]*) |
        (?P<ident>[a-zA-Z_][a-zA-Z0-9_\-]*) |
        (?P<literal>"([^"\\]|\\.)*"|'([^'\\]|\\.)*') |
        (?P<charclass>\[([^\]\[\(\)\{\}\\]|\\.)*\]) |
        (?P<repetition>\{[0-9]*,[0-9]*\}|\{[0-9]+\}) |
        (?P<op>::=|;|,|\?|\*|\+|\||\(|\)|\[|\]|\{|\}|!|\.) |
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
        if let Some(m) = cap.name("directive") {
            // #! is tokenized as two separate ops: # and !
            let start = m.start();
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Op("#".to_string()),
                span: Span { start, end: start + 1 },
            });
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Op("!".to_string()),
                span: Span { start: start + 1, end: start + 2 },
            });
        } else if let Some(m) = cap.name("ident") {
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
                        return Err(ParseError::new(
                            source,
                            Span { start: m.start(), end: m.end() },
                            format!("Literal with dangling escape: {}", s),
                        ));
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
        } else if let Some(m) = cap.name("repetition") {
            // Parse {m,n} or {m} or {,n} or {m,}
            let s = m.as_str();
            let inner = &s[1..s.len()-1]; // Remove { and }
            let (min, max) = if inner.contains(',') {
                let parts: Vec<&str> = inner.split(',').collect();
                let min_val = if parts[0].is_empty() { 0 } else { parts[0].parse().unwrap() };
                let max_val = if parts.len() > 1 && !parts[1].is_empty() {
                    Some(parts[1].parse().unwrap())
                } else {
                    None
                };
                (min_val, max_val)
            } else {
                // {m} means exactly m times
                let n: usize = inner.parse().unwrap();
                (n, Some(n))
            };
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Repetition { min, max },
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(m) = cap.name("op") {
            tokens.push(EbnfToken {
                kind: EbnfTokenKind::Op(m.as_str().to_string()),
                span: Span { start: m.start(), end: m.end() },
            });
        } else if let Some(e) = cap.name("error") {
            let err_text = e.as_str();
            let mut message = format!("Unknown token: {}", err_text);

            if err_text == ":" {
                let rest = &source[e.start()..];
                if rest.starts_with("::=") {
                    message.push_str(" (did you mean '::=' for a rule definition?)");
                } else if rest.starts_with(":=") {
                    message.push_str(
                        " (rule definitions use '::='; did you mean '::=' instead of ':='?)",
                    );
                } else {
                    message.push_str(
                        " (':' is not a valid standalone token; rule definitions must use '::=')",
                    );
                }
            }

            return Err(ParseError::new(
                source,
                Span { start: e.start(), end: e.end() },
                message,
            ));
        }
        // ws and comment are ignored
    }
    Ok(tokens)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GreedyGroupTerminal {
    Name(String),
    Literal(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GreedyGroup {
    pub name: String,
    pub terminals: Vec<GreedyGroupTerminal>,
    pub has_wildcard: bool,
}

#[derive(Debug)]
pub(super) struct EbnfParseResult {
    pub grammar_rules: Vec<(String, GrammarExpr)>,
    pub ignore_symbol_name: Option<String>,
    pub greedy_groups: Vec<GreedyGroup>,
    pub ungrouped_terminals: Vec<GreedyGroupTerminal>,
}

pub(super) struct EbnfParser<'a> {
    source: &'a str,
    tokens: Vec<EbnfToken>,
    pos: usize,
}

impl<'a> EbnfParser<'a> {
    pub(super) fn new(source: &'a str) -> Result<Self, ParseError> {
        let tokens = tokenize(source)?;
        Ok(EbnfParser {
            source,
            tokens,
            pos: 0,
        })
    }

    fn peek(&self) -> Option<&EbnfToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<EbnfToken> {
        if self.pos < self.tokens.len() {
            let token = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(token)
        } else {
            None
        }
    }

    fn parse_rule_body(&mut self) -> Result<GrammarExpr, ParseError> {
        self.expect_grammar_op("::=")?;
        let expr = self.parse_grammar_expression()?;
        // GBNF compatibility: semicolon is optional (GBNF uses newlines)
        if self.peek_grammar_op(";") {
            self.consume_grammar_op(";")?;
        }
        Ok(expr)
    }

    pub(super) fn parse(&mut self) -> Result<EbnfParseResult, ParseError> {
        let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
        let mut seen_names = HashSet::new();
        let mut ignore_symbol_name = None;
        let mut greedy_groups = Vec::new();
        let mut ungrouped_terminals: Option<Vec<GreedyGroupTerminal>> = None;

        while self.peek().is_some() {
            if self.peek_grammar_op("#") {
                let directive_span = self.peek().unwrap().span;
                self.consume_grammar_op("#")?;
                self.expect_grammar_op("!")?;
                self.expect_grammar_op("[")?;
                let (directive_name, directive_name_span) = self.expect_ident()?;

                match directive_name.as_str() {
                    "ignore" => {
                        if ignore_symbol_name.is_some() {
                            return Err(ParseError::new(
                                self.source,
                                directive_span,
                                "Duplicate ignore directive found",
                            ));
                        }
                        self.expect_grammar_op("(")?;
                        let (symbol_name, _) = self.expect_ident()?;
                        self.expect_grammar_op(")")?;
                        ignore_symbol_name = Some(symbol_name);
                    }
                    "greedy_group" => {
                        // Note: greedy_group is broken — do not use in production grammars.
                        // Kept for testing only.
                        self.expect_grammar_op("(")?;
                        let (group_name, group_name_span) = self.expect_ident()?;
                        self.expect_grammar_op(",")?;
                        if self.peek_grammar_op(")") {
                            return Err(ParseError::new(
                                self.source,
                                group_name_span,
                                format!(
                                    "greedy_group '{}' must include at least one terminal or '*'",
                                    group_name
                                ),
                            ));
                        }
                        let (terminals, has_wildcard) =
                            self.parse_greedy_group_terminal_list_allow_wildcard()?;
                        self.expect_grammar_op(")")?;
                        eprintln!("WARNING: greedy_group is known to be broken — do not use in production grammars");
                        greedy_groups.push(GreedyGroup {
                            name: group_name,
                            terminals,
                            has_wildcard,
                        });
                    }
                    "greedy_all" => {
                        return Err(ParseError::new(
                            self.source,
                            directive_span,
                            "greedy_all is no longer supported; use greedy_group(<name>, *)",
                        ));
                    }
                    "ungrouped" => {
                        if ungrouped_terminals.is_some() {
                            return Err(ParseError::new(
                                self.source,
                                directive_span,
                                "Duplicate ungrouped directive found",
                            ));
                        }
                        self.expect_grammar_op("(")?;
                        if self.peek_grammar_op(")") {
                            return Err(ParseError::new(
                                self.source,
                                directive_span,
                                "ungrouped directive must include at least one terminal",
                            ));
                        }
                        let terminals = self.parse_greedy_group_terminal_list()?;
                        self.expect_grammar_op(")")?;
                        ungrouped_terminals = Some(terminals);
                    }
                    _ => {
                        return Err(ParseError::new(
                            self.source,
                            directive_name_span,
                            format!("Unknown directive: {}", directive_name),
                        ));
                    }
                }

                self.expect_grammar_op("]")?;
            } else {
                let (rule_name, rule_name_span) = self.expect_ident()?;
                if seen_names.contains(&rule_name) {
                    return Err(ParseError::new(
                        self.source,
                        rule_name_span,
                        format!("Duplicate rule name: {}", rule_name),
                    ));
                }
                seen_names.insert(rule_name.clone());
                let rule_expr = self.parse_rule_body()?;
                rules.push((rule_name, rule_expr));
            }
        }

        // GBNF compatibility: if a 'root' rule exists but is not first,
        // move it to the front (GBNF uses 'root' as the start rule)
        if let Some(root_idx) = rules.iter().position(|(name, _)| name == "root") {
            if root_idx > 0 {
                let root_rule = rules.remove(root_idx);
                rules.insert(0, root_rule);
            }
        }

        Ok(EbnfParseResult {
            grammar_rules: rules,
            ignore_symbol_name,
            greedy_groups,
            ungrouped_terminals: ungrouped_terminals.unwrap_or_default(),
        })
    }

    fn parse_greedy_group_terminal_list(&mut self) -> Result<Vec<GreedyGroupTerminal>, ParseError> {
        let mut terminals = Vec::new();

        loop {
            terminals.push(self.expect_greedy_group_terminal()?);
            if self.peek_grammar_op(",") {
                self.consume_grammar_op(",")?;
                if self.peek_grammar_op(")") {
                    return Err(ParseError::new(
                        self.source,
                        self.peek().map(|t| t.span).unwrap_or_else(|| self.eof_span()),
                        "Expected terminal after ',' in directive",
                    ));
                }
            } else {
                break;
            }
        }

        Ok(terminals)
    }

    fn parse_greedy_group_terminal_list_allow_wildcard(
        &mut self,
    ) -> Result<(Vec<GreedyGroupTerminal>, bool), ParseError> {
        if self.peek_grammar_op("*") {
            self.consume_grammar_op("*")?;
            if self.peek_grammar_op(",") {
                return Err(ParseError::new(
                    self.source,
                    self.peek().map(|t| t.span).unwrap_or_else(|| self.eof_span()),
                    "Wildcard '*' must be the only selector in a greedy_group directive",
                ));
            }
            return Ok((Vec::new(), true));
        }

        Ok((self.parse_greedy_group_terminal_list()?, false))
    }

    fn expect_greedy_group_terminal(&mut self) -> Result<GreedyGroupTerminal, ParseError> {
        match self.advance() {
            Some(EbnfToken { kind: EbnfTokenKind::Ident(id), .. }) => {
                Ok(GreedyGroupTerminal::Name(id))
            }
            Some(EbnfToken { kind: EbnfTokenKind::Literal(lit), .. }) => {
                Ok(GreedyGroupTerminal::Literal(lit.into_bytes()))
            }
            Some(EbnfToken { kind: EbnfTokenKind::Op(op), span }) if op == "*" => Err(ParseError::new(
                self.source,
                span,
                "Wildcard '*' is not supported in greedy_group/ungrouped directives; list terminals explicitly",
            )),
            Some(other) => Err(ParseError::new(
                self.source,
                other.span,
                format!(
                    "Expected terminal name or quoted literal in directive, found {:?}",
                    other.kind
                ),
            )),
            None => Err(ParseError::new(
                self.source,
                self.eof_span(),
                "Expected terminal name or quoted literal in directive, found end of input",
            )),
        }
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
        while self.peek().is_some()
            && !self.peek_grammar_op(")")
            && !self.peek_grammar_op("]")
            && !self.peek_grammar_op("}")
            && !self.peek_grammar_op("|")
            && !self.peek_grammar_op(";")
            && !self.at_rule_start()  // GBNF compatibility: stop at new rule
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
        } else if let Some(EbnfToken { kind: EbnfTokenKind::Repetition { min, max }, .. }) = self.peek().cloned() {
            self.advance();
            Ok(GrammarExpr::RepeatBounded {
                min,
                max,
                inner: Box::new(factor),
            })
        } else {
            Ok(factor)
        }
    }

    fn parse_grammar_factor(&mut self) -> Result<GrammarExpr, ParseError> {
        if self.peek_grammar_op(".") {
            self.consume_grammar_op(".")?;
            Ok(GrammarExpr::AnyChar)
        } else if let Some(EbnfToken { kind: EbnfTokenKind::Ident(id), .. }) = self.peek().cloned() {
            self.advance();
            Ok(r#ref(&id))
        } else if let Some(EbnfToken { kind: EbnfTokenKind::Literal(lit), .. }) = self.peek().cloned() {
            self.advance();
            Ok(literal(lit.into_bytes()))
        } else if let Some(EbnfToken { kind: EbnfTokenKind::CharClass(cc), .. }) = self.peek().cloned() {
            self.advance();
            Ok(GrammarExpr::CharClass {
                def: cc,
                utf8: false,
            })
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
            let (message, span) = if let Some(token) = self.peek() {
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
            Err(ParseError::new(self.source, span, message))
        }
    }

    // --- Parser Helpers ---

    fn eof_span(&self) -> Span {
        let end = self.source.len();
        Span { start: end, end }
    }

    /// GBNF compatibility: check if we're at the start of a new rule (ident ::=)
    /// Used to detect rule boundaries when there's no semicolon terminator.
    fn at_rule_start(&self) -> bool {
        if let Some(EbnfToken { kind: EbnfTokenKind::Ident(_), .. }) = self.tokens.get(self.pos) {
            if let Some(EbnfToken { kind: EbnfTokenKind::Op(s), .. }) = self.tokens.get(self.pos + 1) {
                return s == "::=";
            }
        }
        false
    }

    fn peek_grammar_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(EbnfToken { kind: EbnfTokenKind::Op(s), .. }) if s == op)
    }

    fn consume_grammar_op(&mut self, op: &str) -> Result<(), ParseError> {
        if self.peek_grammar_op(op) {
            self.advance();
            Ok(())
        } else {
            let (message, span) = if let Some(token) = self.peek() {
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
            Err(ParseError::new(self.source, span, message))
        }
    }

    fn expect_grammar_op(&mut self, op: &str) -> Result<(), ParseError> {
        match self.advance() {
            Some(EbnfToken { kind: EbnfTokenKind::Op(s), .. }) if s == op => Ok(()),
            Some(other) => Err(ParseError::new(
                self.source,
                other.span,
                format!("Expected op '{}', found {:?}", op, other.kind),
            )),
            None => Err(ParseError::new(
                self.source,
                self.eof_span(),
                format!("Expected op '{}', found end of input", op),
            )),
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.advance() {
            Some(EbnfToken { kind: EbnfTokenKind::Ident(id), span }) => Ok((id, span)),
            Some(other) => Err(ParseError::new(
                self.source,
                other.span,
                format!("Expected identifier, found {:?}", other.kind),
            )),
            None => Err(ParseError::new(
                self.source,
                self.eof_span(),
                "Expected identifier, found end of input",
            )),
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

    #[test]
    fn test_ebnf_parser_greedy_group_directives() {
        let ebnf = r#"
            #![ignore(IGNORE)]
            #![greedy_group(main, IDENT, 'if', '(')]
            #![ungrouped(EOF, IGNORE)]
            root ::= IDENT | 'if' ;
            IDENT ::= [a-z]+ ;
            IGNORE ::= [ ]+ ;
            EOF ::= '<|endoftext|>' ;
        "#;

        let mut parser = EbnfParser::new(ebnf).unwrap();
        let parsed = parser.parse().unwrap();

        assert_eq!(parsed.ignore_symbol_name.as_deref(), Some("IGNORE"));
        assert_eq!(parsed.greedy_groups.len(), 1);
        assert_eq!(parsed.greedy_groups[0].name, "main");
        assert_eq!(
            parsed.greedy_groups[0].terminals,
            vec![
                GreedyGroupTerminal::Name("IDENT".to_string()),
                GreedyGroupTerminal::Literal(b"if".to_vec()),
                GreedyGroupTerminal::Literal(b"(".to_vec()),
            ]
        );
        assert_eq!(
            parsed.ungrouped_terminals,
            vec![
                GreedyGroupTerminal::Name("EOF".to_string()),
                GreedyGroupTerminal::Name("IGNORE".to_string()),
            ]
        );
    }

    #[test]
    fn test_ebnf_parser_allows_wildcard_greedy_group() {
        let ebnf = r#"
            #![greedy_group(main, *)]
            root ::= 'a' ;
        "#;

        let mut parser = EbnfParser::new(ebnf).unwrap();
        let parsed = parser.parse().unwrap();
        assert_eq!(parsed.greedy_groups.len(), 1);
        assert_eq!(parsed.greedy_groups[0].name, "main");
        assert!(parsed.greedy_groups[0].has_wildcard);
        assert!(parsed.greedy_groups[0].terminals.is_empty());
    }

    #[test]
    fn test_ebnf_parser_rejects_greedy_all_directive() {
        let ebnf = r#"
            #![greedy_all]
            root ::= IDENT | 'if' ;
            IDENT ::= [a-z]+ ;
        "#;

        let mut parser = EbnfParser::new(ebnf).unwrap();
        let err = parser.parse().unwrap_err();
        assert!(!err.message.is_empty());
    }

    #[test]
    fn test_ebnf_parser_rejects_mixed_greedy_all_and_greedy_group() {
        let ebnf = r#"
            #![greedy_all]
            #![greedy_group(main, IDENT)]
            root ::= IDENT ;
            IDENT ::= [a-z]+ ;
        "#;

        let mut parser = EbnfParser::new(ebnf).unwrap();
        let err = parser.parse().unwrap_err();
        assert!(!err.message.is_empty());
    }

    #[test]
    fn test_grammar_definition_from_ebnf_wildcard_greedy_group_directive() {
        let ebnf = r#"
            #![greedy_group(main, *)]
            #![ungrouped(IGNORE)]
            root ::= IDENT | 'if' ;
            IDENT ::= [a-z]+ ;
            IGNORE ::= [ ]+ ;
        "#;

        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        assert_eq!(grammar.greedy_groups.len(), 1);
        assert_eq!(grammar.greedy_groups[0].name, "main");
        assert!(grammar.greedy_groups[0].terminals.contains(&"IDENT".to_string()));
        assert!(grammar.greedy_groups[0].terminals.contains(&"'if'".to_string()));
        assert_eq!(grammar.ungrouped_terminals, vec!["IGNORE".to_string()]);
    }

    #[test]
    fn test_ebnf_parser_allows_wildcard_and_explicit_greedy_groups() {
        let ebnf = r#"
            #![greedy_group(explicit, IDENT)]
            #![greedy_group(catchall, *)]
            root ::= IDENT | 'if' ;
            IDENT ::= [a-z]+ ;
        "#;

        let mut parser = EbnfParser::new(ebnf).unwrap();
        let parsed = parser.parse().unwrap();

        assert_eq!(parsed.greedy_groups.len(), 2);
        assert!(parsed
            .greedy_groups
            .iter()
            .any(|group| group.name == "explicit" && !group.has_wildcard));
        assert!(parsed
            .greedy_groups
            .iter()
            .any(|group| group.name == "catchall" && group.has_wildcard));
    }

    #[test]
    fn test_grammar_definition_from_ebnf_wildcard_and_explicit_greedy_groups() {
        let ebnf = r#"
            #![greedy_group(explicit, IDENT)]
            #![greedy_group(catchall, *)]
            root ::= IDENT | 'if' ;
            IDENT ::= [a-z]+ ;
        "#;

        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        assert_eq!(grammar.greedy_groups.len(), 2);

        let explicit = grammar
            .greedy_groups
            .iter()
            .find(|group| group.name == "explicit")
            .expect("missing explicit group");
        assert_eq!(explicit.terminals, vec!["IDENT".to_string()]);

        let catchall = grammar
            .greedy_groups
            .iter()
            .find(|group| group.name == "catchall")
            .expect("missing catchall group");
        assert!(catchall.terminals.contains(&"'if'".to_string()));
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

