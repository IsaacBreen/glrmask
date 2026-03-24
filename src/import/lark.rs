#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
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
        GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
            parts
                .iter()
                .map(|part| {
                    expand_lark_expr(
                        part,
                        in_terminal_rule,
                        rule_map,
                        terminal_names,
                        parser_names,
                        memo,
                        visiting,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        GrammarExpr::Choice(options) => GrammarExpr::Choice(
            options
                .iter()
                .map(|option| {
                    expand_lark_expr(
                        option,
                        in_terminal_rule,
                        rule_map,
                        terminal_names,
                        parser_names,
                        memo,
                        visiting,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(expand_lark_expr(
                expr,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?),
            exclude: Box::new(expand_lark_expr(
                exclude,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?),
        },
        GrammarExpr::Optional(inner) => GrammarExpr::Optional(Box::new(expand_lark_expr(
            inner,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?)),
        GrammarExpr::Repeat(inner) => GrammarExpr::Repeat(Box::new(expand_lark_expr(
            inner,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?)),
        GrammarExpr::RepeatOne(inner) => GrammarExpr::RepeatOne(Box::new(expand_lark_expr(
            inner,
            in_terminal_rule,
            rule_map,
            terminal_names,
            parser_names,
            memo,
            visiting,
        )?)),
        GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
            expr: Box::new(expand_lark_expr(
                expr,
                in_terminal_rule,
                rule_map,
                terminal_names,
                parser_names,
                memo,
                visiting,
            )?),
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

    // Validate invariant: inside terminal definitions, every Ref must
    // resolve to another terminal (not a nonterminal, and not missing).
    fn validate_terminal_refs(
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
                for p in parts {
                    validate_terminal_refs(p, rule_name, terminal_names, rule_map)?;
                }
                Ok(())
            }
            GrammarExpr::Exclude { expr, exclude } => {
                validate_terminal_refs(expr, rule_name, terminal_names, rule_map)?;
                validate_terminal_refs(exclude, rule_name, terminal_names, rule_map)
            }
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner)
            | GrammarExpr::RepeatRange { expr: inner, .. } => {
                validate_terminal_refs(inner, rule_name, terminal_names, rule_map)
            }
            _ => Ok(()),
        }
    }

    for rule in &grammar.rules {
        if terminal_names.contains(&rule.name) {
            validate_terminal_refs(&rule.expr, &rule.name, &terminal_names, &rule_map)?;
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

    // Step 1: Expand each Lark terminal rule (inlining sub-terminal refs).
    // The expanded GrammarExpr is stored directly — lowering converts it
    // to an Expr tree without a string intermediate.
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
        rules.push(NamedRule { name: rule.name.clone(), expr: expanded, is_terminal: true });
    }

    // Step 2: Process parser rules.  Terminal references stay as Ref nodes
    // so lower() resolves them to the nonterminal wrappers created above.
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
        rules.push(NamedRule { name: rule.name.clone(), expr: expanded, is_terminal: false });
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
            rules.insert(0, NamedRule { name: output_start.clone(), expr: start_expr, is_terminal: true });
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

    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        let mut rules: Vec<NamedRule> = Vec::new();
        let mut ignore_exprs: Vec<GrammarExpr> = Vec::new();

        self.skip_newlines();
        while self.pos < self.tokens.len() {
            // Handle %ignore directives
            if self.peek() == Some(&Token::PercentIgnore) {
                self.pos += 1;
                let expr = self.parse_atom()?;
                ignore_exprs.push(expr);
                self.skip_newlines();
                continue;
            }

            while matches!(self.peek(), Some(Token::Question) | Some(Token::Bang)) {
                self.pos += 1;
            }

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

            if self.peek() == Some(&Token::Dot)
                && matches!(self.peek_nth(1), Some(Token::Number(_)))
            {
                self.pos += 2;
            }

            self.expect_token(&Token::Colon)?;

            let expr = self.parse_alternatives()?;

            rules.push(NamedRule {
                name,
                expr,
                is_terminal: false, // set correctly below
            });
            self.skip_newlines();
        }

        if rules.is_empty() {
            return Err(GlrMaskError::GrammarParse("empty grammar".into()));
        }

        let start = if rules.iter().any(|r| r.name == "start") {
            "start".to_string()
        } else {
            rules[0].name.clone()
        };

        // Mark terminal rules based on Lark naming convention
        for rule in &mut rules {
            rule.is_terminal = is_lark_terminal_name(&rule.name);
        }

        // Synthesize an __IGNORE terminal if %ignore directives were found
        let ignore = if ignore_exprs.is_empty() {
            None
        } else {
            let ignore_body = if ignore_exprs.len() == 1 {
                ignore_exprs.into_iter().next().unwrap()
            } else {
                GrammarExpr::Choice(ignore_exprs)
            };
            let name = "__IGNORE".to_string();
            rules.push(NamedRule { name: name.clone(), expr: ignore_body, is_terminal: true });
            Some(name)
        };

        Ok(NamedGrammar { rules, start, ignore })
    }

    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let first = self.parse_sequence()?;
        self.consume_alias_if_present()?;
        let mut alts = vec![first];

        loop {
            
            if self.peek() == Some(&Token::Pipe) {
                self.pos += 1;
                let alt = self.parse_sequence()?;
                self.consume_alias_if_present()?;
                alts.push(alt);
                continue;
            }
            
            let saved = self.pos;
            let mut saw_newline = false;
            while self.peek() == Some(&Token::Newline) {
                self.pos += 1;
                saw_newline = true;
            }
            if saw_newline && self.peek() == Some(&Token::Pipe) {
                self.pos += 1;
                let alt = self.parse_sequence()?;
                self.consume_alias_if_present()?;
                alts.push(alt);
                continue;
            }
            
            self.pos = saved;
            break;
        }

        if alts.len() == 1 {
            Ok(alts.into_iter().next().unwrap())
        } else {
            Ok(GrammarExpr::Choice(alts))
        }
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
                self.pos += 1;
                let min = match self.advance() {
                    Some(Token::Number(n)) => n,
                    _ => return Err(GlrMaskError::GrammarParse("expected number after ~".into())),
                };
                
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
                        
                        self.pos = saved;
                        None
                    }
                } else {
                    None
                };
                Ok(bounded_repeat_expr(atom, min, max))
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(s)) => {
                if self.peek() == Some(&Token::Dot) && self.peek_nth(1) == Some(&Token::Dot) {
                    self.pos += 2;
                    match self.advance() {
                        Some(Token::Literal(end)) => return literal_range_expr(&s, &end),
                        Some(other) => {
                            return Err(GlrMaskError::GrammarParse(format!(
                                "expected literal after .. in Lark literal range, got {:?}",
                                other
                            )))
                        }
                        None => {
                            return Err(GlrMaskError::GrammarParse(
                                "expected literal after .. in Lark literal range, got end of input"
                                    .into(),
                            ))
                        }
                    }
                }

                if s.is_empty() {
                    Ok(GrammarExpr::Sequence(vec![]))
                } else {
                    Ok(GrammarExpr::Literal(s.into_bytes()))
                }
            }
            Some(Token::Regex(rx)) => {
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

pub fn parse_lark(input: &str) -> Result<GrammarDef, GlrMaskError> {
    let named = parse_lark_to_named(input)?;
    let factored = factor_named_grammar(named);
    lower(&factored)
}

#[allow(dead_code)]
    
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
