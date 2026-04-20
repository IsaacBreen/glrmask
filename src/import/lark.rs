use std::collections::{HashMap, HashSet};

use crate::GlrMaskError;
use crate::grammar::flat::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, lower};
use crate::grammar::factoring::factor_named_grammar;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Terminal(String),
    Literal(String),
    Regex(String),
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
    Tilde,
    Number(usize),
    Comma,
    Arrow,
    Bang,
    PercentIgnore,
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
                    Some(c) if c == quote => s.push(c as char),
                    Some(b'"') => s.push('"'),
                    Some(b'\'') => s.push('\''),
                    Some(b'x') => {
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
                    self.pos += 1;
                    // Check for %ignore directive
                    let rest = &self.input[self.pos..];
                    if rest.starts_with(b"ignore") && rest.get(6).map_or(true, |b| !b.is_ascii_alphanumeric() && *b != b'_') {
                        self.pos += 6;
                        tokens.push(Token::PercentIgnore);
                    } else {
                        // Skip other %directives (e.g. %import)
                        self.skip_comment();
                    }
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
                Some(b'\'') => {
                    self.pos += 1;
                    let s = self.lex_string(b'\'')?;
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
                Some(b'!') => {
                    self.pos += 1;
                    tokens.push(Token::Bang);
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

fn bounded_repeat_expr(atom: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
    let max = max.unwrap_or(min);
    assert!(max >= min, "tilde max must be >= min");
    GrammarExpr::RepeatRange {
        expr: Box::new(atom),
        min,
        max,
    }
}

fn choice_or_single(mut options: Vec<GrammarExpr>) -> GrammarExpr {
    if options.len() == 1 {
        options.pop().unwrap()
    } else {
        GrammarExpr::Choice(options)
    }
}

fn sequence_or_single(mut parts: Vec<GrammarExpr>) -> GrammarExpr {
    match parts.len() {
        0 => GrammarExpr::Sequence(Vec::new()),
        1 => parts.pop().unwrap(),
        _ => GrammarExpr::Sequence(parts),
    }
}

fn escape_char_class_byte(b: u8) -> String {
    match b {
        b'\\' | b']' | b'^' | b'-' => format!("\\{}", b as char),
        b'\n' => "\\n".into(),
        b'\r' => "\\r".into(),
        b'\t' => "\\t".into(),
        byte if byte.is_ascii_graphic() || byte == b' ' => (byte as char).to_string(),
        byte => format!("\\x{byte:02x}"),
    }
}

fn literal_range_expr(start: &str, end: &str) -> Result<GrammarExpr, GlrMaskError> {
    let start_bytes = start.as_bytes();
    let end_bytes = end.as_bytes();
    if start_bytes.len() != 1 || end_bytes.len() != 1 {
        return Err(GlrMaskError::GrammarParse(
            "Lark literal ranges currently require single-byte endpoints".into(),
        ));
    }

    let start_byte = start_bytes[0];
    let end_byte = end_bytes[0];
    if start_byte > end_byte {
        return Err(GlrMaskError::GrammarParse(format!(
            "invalid Lark literal range {:?}..{:?}",
            start, end
        )));
    }

    Ok(GrammarExpr::CharClass {
        def: format!(
            "{}-{}",
            escape_char_class_byte(start_byte),
            escape_char_class_byte(end_byte)
        ),
        negate: false,
        utf8: true,
    })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

fn is_lark_terminal_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

fn lark_start_rule_name(rules: &[NamedRule]) -> String {
    if rules.iter().any(|rule| rule.name == "start") {
        "start".to_string()
    } else {
        rules[0].name.clone()
    }
}

fn mark_lark_terminal_rules(rules: &mut [NamedRule]) {
    for rule in rules {
        rule.is_terminal = is_lark_terminal_name(&rule.name);
    }
}

fn synthesize_lark_ignore_rule(
    rules: &mut Vec<NamedRule>,
    ignore_exprs: Vec<GrammarExpr>,
) -> Option<String> {
    if ignore_exprs.is_empty() {
        return None;
    }

    let ignore_name = "__IGNORE".to_string();
    rules.push(NamedRule {
        name: ignore_name.clone(),
        expr: choice_or_single(ignore_exprs),
        is_terminal: true,
        is_internal: false,
    });
    Some(ignore_name)
}

fn expand_lark_expr_list(
    exprs: &[GrammarExpr],
    in_terminal_rule: bool,
    rule_map: &HashMap<String, GrammarExpr>,
    terminal_names: &HashSet<String>,
    parser_names: &HashSet<String>,
    memo: &mut HashMap<String, GrammarExpr>,
    visiting: &mut HashSet<String>,
) -> Result<Vec<GrammarExpr>, GlrMaskError> {
    exprs
        .iter()
        .map(|expr| {
            expand_lark_expr(
                expr,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )
        })
        .collect()
}

fn expand_lark_boxed_expr(
    expr: &GrammarExpr,
    in_terminal_rule: bool,
    rule_map: &HashMap<String, GrammarExpr>,
    terminal_names: &HashSet<String>,
    parser_names: &HashSet<String>,
    memo: &mut HashMap<String, GrammarExpr>,
    visiting: &mut HashSet<String>,
) -> Result<Box<GrammarExpr>, GlrMaskError> {
    Ok(Box::new(expand_lark_expr(
        expr,
        in_terminal_rule,
        rule_map,
        terminal_names,
        parser_names,
        memo,
        visiting,
    )?))
}

fn validate_lark_terminal_refs(
    expr: &GrammarExpr,
    rule_name: &str,
    terminal_names: &HashSet<String>,
    rule_map: &HashMap<String, GrammarExpr>,
) -> Result<(), GlrMaskError> {
    match expr {
        GrammarExpr::Ref(target) => {
            if !rule_map.contains_key(target) {
                return Err(GlrMaskError::GrammarParse(format!(
                    "terminal rule {rule_name} references undefined rule {target}"
                )));
            }
            if !terminal_names.contains(target) {
                return Err(GlrMaskError::GrammarParse(format!(
                    "terminal rule {rule_name} references nonterminal {target}"
                )));
            }
            Ok(())
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            for part in parts {
                validate_lark_terminal_refs(part, rule_name, terminal_names, rule_map)?;
            }
            Ok(())
        }
        GrammarExpr::Exclude { expr, exclude } => {
            validate_lark_terminal_refs(expr, rule_name, terminal_names, rule_map)?;
            validate_lark_terminal_refs(exclude, rule_name, terminal_names, rule_map)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => {
            validate_lark_terminal_refs(inner, rule_name, terminal_names, rule_map)
        }
        _ => Ok(()),
    }
}

fn expand_lark_terminal_rule(
    name: &str,
    rule_map: &HashMap<String, GrammarExpr>,
    terminal_names: &HashSet<String>,
    parser_names: &HashSet<String>,
    memo: &mut HashMap<String, GrammarExpr>,
    visiting: &mut HashSet<String>,
) -> Result<GrammarExpr, GlrMaskError> {
    if let Some(cached) = memo.get(name) {
        return Ok(cached.clone());
    }

    if !visiting.insert(name.to_string()) {
        return Err(GlrMaskError::GrammarParse(format!(
            "cyclic Lark terminal definition involving {name}"
        )));
    }

    let expr = rule_map.get(name).ok_or_else(|| {
        GlrMaskError::GrammarParse(format!("unknown Lark terminal rule {name}"))
    })?;
    let expanded = expand_lark_expr(
        expr,
        true,
        rule_map,
        terminal_names,
        parser_names,
        memo,
        visiting,
    )?;
    visiting.remove(name);
    memo.insert(name.to_string(), expanded.clone());
    Ok(expanded)
}

fn expand_lark_expr(
    expr: &GrammarExpr,
    in_terminal_rule: bool,
    rule_map: &HashMap<String, GrammarExpr>,
    terminal_names: &HashSet<String>,
    parser_names: &HashSet<String>,
    memo: &mut HashMap<String, GrammarExpr>,
    visiting: &mut HashSet<String>,
) -> Result<GrammarExpr, GlrMaskError> {
    Ok(match expr {
        GrammarExpr::Ref(name) => {
            if terminal_names.contains(name) {
                if in_terminal_rule {
                    // Inside a terminal definition, expand referenced terminal
                    expand_lark_terminal_rule(name, rule_map, terminal_names, parser_names, memo, visiting)?
                } else {
                    // In a parser rule, keep the reference (resolved in lower())
                    GrammarExpr::Ref(name.clone())
                }
            } else if parser_names.contains(name) {
                if in_terminal_rule {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "Lark terminal rule cannot reference parser rule {name}"
                    )));
                }
                GrammarExpr::Ref(name.clone())
            } else {
                return Err(GlrMaskError::GrammarParse(format!(
                    "unknown Lark rule reference {name}"
                )));
            }
        }
        GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(expand_lark_expr_list(
            parts,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?),
        GrammarExpr::Choice(options) => GrammarExpr::Choice(expand_lark_expr_list(
            options,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: expand_lark_boxed_expr(
                expr,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?,
            exclude: expand_lark_boxed_expr(
                exclude,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?,
        },
        GrammarExpr::Optional(inner) => GrammarExpr::Optional(expand_lark_boxed_expr(
            inner,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?),
        GrammarExpr::Repeat(inner) => GrammarExpr::Repeat(expand_lark_boxed_expr(
            inner,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?),
        GrammarExpr::RepeatOne(inner) => GrammarExpr::RepeatOne(expand_lark_boxed_expr(
            inner,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?),
        GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
            expr: expand_lark_boxed_expr(
                expr,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?,
            min: *min,
            max: *max,
        },
        GrammarExpr::Literal(bytes) => GrammarExpr::Literal(bytes.clone()),
        GrammarExpr::CharClass { def, negate, utf8 } => GrammarExpr::CharClass {
            def: def.clone(),
            negate: *negate,
            utf8: *utf8,
        },
        GrammarExpr::RawRegex(pattern) => GrammarExpr::RawRegex(pattern.clone()),
        GrammarExpr::TerminalExpr(expr) => GrammarExpr::TerminalExpr(expr.clone()),
        GrammarExpr::AnyByte => GrammarExpr::AnyByte,
        GrammarExpr::SeparatedSequence { items, separator } => {
            let new_items = items
                .iter()
                .map(|(item, req)| {
                    Ok((
                        expand_lark_expr(
                            item,
                            in_terminal_rule,
                            rule_map,
                            terminal_names,
                            parser_names,
                            memo,
                            visiting,
                        )?,
                        *req,
                    ))
                })
                .collect::<Result<Vec<_>, GlrMaskError>>()?;
            let new_separator = expand_lark_expr(
                separator,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?;
            GrammarExpr::SeparatedSequence {
                items: new_items,
                separator: Box::new(new_separator),
            }
        }
    })
}

fn normalize_lark_named(grammar: NamedGrammar) -> Result<NamedGrammar, GlrMaskError> {
    let rule_map: HashMap<String, GrammarExpr> = grammar.rules.iter().map(|r| (r.name.clone(), r.expr.clone())).collect();
    let terminal_names: HashSet<String> = grammar.terminal_names_set();
    let parser_names: HashSet<String> = grammar
        .rules
        .iter()
        .map(|r| r.name.clone())
        .filter(|name| !terminal_names.contains(name))
        .collect();

    for rule in &grammar.rules {
        if terminal_names.contains(&rule.name) {
            validate_lark_terminal_refs(&rule.expr, &rule.name, &terminal_names, &rule_map)?;
        }
    }

    let mut memo = HashMap::new();
    let mut visiting = HashSet::new();
    let mut rules = Vec::new();

    let start_is_terminal = terminal_names.contains(&grammar.start);
    let output_start = if start_is_terminal {
        "start".to_string()
    } else {
        grammar.start.clone()
    };

    // Expand each Lark terminal rule and keep the lowered expression directly.
    for rule in &grammar.rules {
        if !terminal_names.contains(&rule.name) {
            continue;
        }
        let expanded = expand_lark_terminal_rule(
            &rule.name,
            &rule_map,
            &terminal_names,
            &parser_names,
            &mut memo,
            &mut visiting,
        )?;
        rules.push(NamedRule { name: rule.name.clone(), expr: expanded, is_terminal: true, is_internal: false });
    }

    // Process parser rules while leaving terminal references as `Ref` nodes.
    for rule in &grammar.rules {
        if terminal_names.contains(&rule.name) {
            continue;
        }
        let expanded = expand_lark_expr(
            &rule.expr,
            false,
            &rule_map,
            &terminal_names,
            &parser_names,
            &mut memo,
            &mut visiting,
        )?;
        rules.push(NamedRule { name: rule.name.clone(), expr: expanded, is_terminal: false, is_internal: false });
    }

    if start_is_terminal {
        let start_expr = expand_lark_terminal_rule(
            &grammar.start,
            &rule_map,
            &terminal_names,
            &parser_names,
            &mut memo,
            &mut visiting,
        )?;
        if let Some(existing) = rules.iter_mut().find(|r| r.name == output_start) {
            existing.expr = start_expr;
        } else {
            rules.insert(0, NamedRule { name: output_start.clone(), expr: start_expr, is_terminal: true, is_internal: false });
        }
    }

    Ok(NamedGrammar {
        rules,
        start: output_start,
        ignore: grammar.ignore,
    })
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_nth(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n)
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

    fn parse_rule_name(&mut self) -> Result<String, GlrMaskError> {
        while matches!(self.peek(), Some(Token::Question) | Some(Token::Bang)) {
            self.pos += 1;
        }

        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(name),
            Some(other) => Err(GlrMaskError::GrammarParse(format!(
                "expected rule name, got {:?}",
                other
            ))),
            None => Err(GlrMaskError::GrammarParse(
                "expected rule name, got end of input".into(),
            )),
        }
    }

    fn skip_rule_priority(&mut self) {
        if self.peek() == Some(&Token::Dot) && matches!(self.peek_nth(1), Some(Token::Number(_))) {
            self.pos += 2;
        }
    }

    fn parse_bounded_repeat(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        let min = match self.advance() {
            Some(Token::Number(value)) => value,
            _ => return Err(GlrMaskError::GrammarParse("expected number after ~".into())),
        };

        let max = if self.peek() == Some(&Token::Dot) {
            let saved = self.pos;
            self.pos += 1;
            if self.peek() == Some(&Token::Dot) {
                self.pos += 1;
                match self.advance() {
                    Some(Token::Number(value)) => Some(value),
                    _ => {
                        return Err(GlrMaskError::GrammarParse(
                            "expected number after ..".into(),
                        ));
                    }
                }
            } else {
                self.pos = saved;
                None
            }
        } else {
            None
        };

        Ok(bounded_repeat_expr(atom, min, max))
    }

    fn parse_literal_or_range(&mut self, literal: String) -> Result<GrammarExpr, GlrMaskError> {
        if self.peek() == Some(&Token::Dot) && self.peek_nth(1) == Some(&Token::Dot) {
            self.pos += 2;
            return match self.advance() {
                Some(Token::Literal(end)) => literal_range_expr(&literal, &end),
                Some(other) => Err(GlrMaskError::GrammarParse(format!(
                    "expected literal after .. in Lark literal range, got {:?}",
                    other
                ))),
                None => Err(GlrMaskError::GrammarParse(
                    "expected literal after .. in Lark literal range, got end of input".into(),
                )),
            };
        }

        if literal.is_empty() {
            Ok(sequence_or_single(Vec::new()))
        } else {
            Ok(GrammarExpr::Literal(literal.into_bytes()))
        }
    }

    fn parse_ignore_directive(
        &mut self,
        ignore_exprs: &mut Vec<GrammarExpr>,
    ) -> Result<bool, GlrMaskError> {
        if self.peek() != Some(&Token::PercentIgnore) {
            return Ok(false);
        }

        self.pos += 1;
        ignore_exprs.push(self.parse_atom()?);
        self.skip_newlines();
        Ok(true)
    }

    fn parse_rule(&mut self) -> Result<NamedRule, GlrMaskError> {
        let name = self.parse_rule_name()?;
        self.skip_rule_priority();
        self.expect_token(&Token::Colon)?;
        let expr = self.parse_alternatives()?;
        Ok(NamedRule {
            name,
            expr,
            is_terminal: false,
            is_internal: false,
        })
    }

    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        let mut rules: Vec<NamedRule> = Vec::new();
        let mut ignore_exprs: Vec<GrammarExpr> = Vec::new();

        self.skip_newlines();
        while self.pos < self.tokens.len() {
            if self.parse_ignore_directive(&mut ignore_exprs)? {
                continue;
            }

            rules.push(self.parse_rule()?);
            self.skip_newlines();
        }

        if rules.is_empty() {
            return Err(GlrMaskError::GrammarParse("empty grammar".into()));
        }

        let start = lark_start_rule_name(&rules);
        mark_lark_terminal_rules(&mut rules);
        let ignore = synthesize_lark_ignore_rule(&mut rules, ignore_exprs);

        Ok(NamedGrammar { rules, start, ignore })
    }

    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let first = self.parse_sequence()?;
        self.consume_alias_if_present()?;
        let mut alts = vec![first];

        while self.consume_alternative_separator() {
            let alt = self.parse_sequence()?;
            self.consume_alias_if_present()?;
            alts.push(alt);
        }

        Ok(choice_or_single(alts))
    }

    fn consume_alternative_separator(&mut self) -> bool {
        if self.peek() == Some(&Token::Pipe) {
            self.pos += 1;
            return true;
        }

        let saved = self.pos;
        while self.peek() == Some(&Token::Newline) {
            self.pos += 1;
        }
        if self.pos > saved && self.peek() == Some(&Token::Pipe) {
            self.pos += 1;
            return true;
        }

        self.pos = saved;
        false
    }

    fn consume_alias_if_present(&mut self) -> Result<(), GlrMaskError> {
        if self.peek() != Some(&Token::Arrow) {
            return Ok(());
        }

        self.pos += 1;
        match self.advance() {
            Some(Token::Ident(_)) | Some(Token::Terminal(_)) => Ok(()),
            Some(other) => Err(GlrMaskError::GrammarParse(format!(
                "expected alias name after ->, got {:?}",
                other
            ))),
            None => Err(GlrMaskError::GrammarParse(
                "expected alias name after ->, got end of input".into(),
            )),
        }
    }

    fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let mut parts = Vec::new();

        while self.is_unit_start() {
            parts.push(self.parse_unit()?);
        }

        Ok(sequence_or_single(parts))
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
                self.pos += 1;
                self.parse_bounded_repeat(atom)
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(literal)) => self.parse_literal_or_range(literal),
            Some(Token::Regex(regex)) => Ok(GrammarExpr::RawRegex(regex)),
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

pub fn parse_lark(input: &str) -> Result<GrammarDef, GlrMaskError> {
    let named = parse_lark_to_named(input)?;
    let factored = factor_named_grammar(named);
    lower(&factored)
}

pub fn parse_lark_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let named = parser.parse_grammar()?;
    normalize_lark_named(named)
}

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

    #[test]
    fn test_parse_single_quoted_literals_and_aliases() {
        let g = parse_lark("start: 'a' -> left | \"b\" -> right").unwrap();
        let start_rules: Vec<_> = g.rules.iter().filter(|r| r.lhs == g.start).collect();
        assert_eq!(start_rules.len(), 2);
        assert_eq!(g.num_terminals(), 2);
    }

    #[test]
    fn test_parse_literal_range_terminal() {
        let g = parse_lark("start: DIGIT\nDIGIT: '0'..'9'").unwrap();
        assert_eq!(g.num_terminals(), 1);
    }

    #[test]
    fn test_parse_rule_prefix_and_priority() {
        let g = parse_lark("?start: ATOM\nATOM.2: 'a'").unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_lark_terminal_rules_are_preserved_as_composite_units() {
        let named = parse_lark_to_named("start: WORD\nWORD: LETTER+\nLETTER: 'a' | 'b'").unwrap();
        // Terminal rules are stored as expanded GrammarExpr trees (no Ref nodes).
        // lower() converts them directly to Expr using grammar.terminals.
        assert_eq!(named.rules.len(), 3);
        assert_eq!(named.rules[0].name, "WORD");
        // WORD = RepeatOne(Choice([Literal("a"), Literal("b")]))
        assert_eq!(
            named.rules[0].expr,
            GrammarExpr::RepeatOne(Box::new(GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"a".to_vec()),
                GrammarExpr::Literal(b"b".to_vec()),
            ])))
        );
        assert_eq!(named.rules[1].name, "LETTER");
        // LETTER = Choice([Literal("a"), Literal("b")])
        assert_eq!(
            named.rules[1].expr,
            GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"a".to_vec()),
                GrammarExpr::Literal(b"b".to_vec()),
            ])
        );
        assert_eq!(named.rules[2].name, "start");
        assert_eq!(named.rules[2].expr, GrammarExpr::Ref("WORD".into()));
    }

    #[test]
    fn test_lark_terminal_rule_cannot_reference_parser_rule() {
        let err = parse_lark_to_named("start: WORD\nitem: 'a'\nWORD: item")
            .expect_err("terminal rule referencing parser rule should fail");
        assert!(
            err.to_string().contains("references nonterminal item"),
            "unexpected error: {err}"
        );
    }
}
