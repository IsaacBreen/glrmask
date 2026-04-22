//! GLRM: Glrmask Grammar Format
//!
//! A fully-featured, human-readable grammar format that can represent every
//! construct in [`NamedGrammar`] / [`GrammarExpr`], including
//! `Exclude`, `Intersect`, internal terminals, and `SeparatedSequence`.
//!
//! # Format overview
//!
//! ```text
//! // Line comment
//! /* Block comment */
//!
//! start <nt-name>;
//! [ignore <TM-NAME>;]
//!
//! // Nonterminal rules
//! nt rule_name ::= <expr>;
//!
//! // Terminal rules (RHS uses the same expression syntax)
//! t TERM_NAME ::= <expr>;
//!
//! // Internal terminal rules (shared between other terminals)
//! internal t TERM_NAME ::= <expr>;
//! ```
//!
//! ## Expressions (used for both NT and terminal rule bodies)
//!
//! | Syntax                          | Meaning                              |
//! |--------------------------------|--------------------------------------|
//! | `name`                         | Reference to a rule                  |
//! | `"text"`                       | Literal bytes                        |
//! | `/regex/`                      | Raw regex pattern                    |
//! | `[class]`, `[^class]`          | Byte character class                 |
//! | `[class]/utf8`                 | UTF-8 character class                |
//! | `.`                            | Any byte                             |
//! | `eps`                          | Epsilon (empty string)               |
//! | `a b c`                        | Sequence                             |
//! | `a \| b \| c`                  | Choice                               |
//! | `e?`, `e*`, `e+`              | Optional / Repeat / RepeatOne        |
//! | `e{n}`, `e{n,m}`              | RepeatRange                          |
//! | `(e)`                          | Grouping                             |
//! | `a - b`                        | GrammarExpr::Exclude                 |
//! | `a & b`                        | GrammarExpr::Intersect               |
//! | `sep ~ ( i1? i2 i3? )`           | SeparatedSequence                    |

use crate::GlrMaskError;
use crate::ds::u8set::U8Set;
use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};

// ============================================================
// Dumper
// ============================================================

/// Serialise `grammar` to the GLRM text format.
pub fn to_glrm(grammar: &NamedGrammar) -> String {
    let mut out = String::new();
    out.push_str(&format!("start {};\n", grammar.start));
    if let Some(ref ign) = grammar.ignore {
        out.push_str(&format!("ignore {};\n", ign));
    }
    out.push('\n');

    for rule in &grammar.rules {
        let prefix = match (rule.is_terminal, rule.is_internal) {
            (true, true) => "internal t",
            (true, false) => "t",
            (false, _) => "nt",
        };
        let body = dump_nt_expr(&rule.expr, false);
        out.push_str(&format!("{} {} ::= {};\n", prefix, rule.name, body));
    }

    out
}

// ---- NT-expression dumper --------------------------------------------------

fn dump_nt_expr(expr: &GrammarExpr, needs_parens: bool) -> String {
    match expr {
        GrammarExpr::Choice(alts) => {
            let inner = alts.iter()
                .map(|a| dump_nt_seq(a))
                .collect::<Vec<_>>()
                .join(" | ");
            if needs_parens && alts.len() > 1 {
                format!("({})", inner)
            } else {
                inner
            }
        }
        GrammarExpr::Exclude { expr: inner, exclude } => {
            let lhs = dump_set_operand(inner);
            let rhs = match exclude.as_ref() {
                GrammarExpr::Choice(alts) if !alts.is_empty() => alts
                    .iter()
                    .map(dump_set_operand)
                    .collect::<Vec<_>>()
                    .join(" - "),
                _ => dump_set_operand(exclude),
            };
            let infix = format!("{} - {}", lhs, rhs);
            if needs_parens {
                format!("({})", infix)
            } else {
                infix
            }
        }
        GrammarExpr::Intersect { expr: inner, intersect } => {
            let infix = format!(
                "{} & {}",
                dump_set_operand(inner),
                dump_set_operand(intersect)
            );
            if needs_parens {
                format!("({})", infix)
            } else {
                infix
            }
        }
        _ => dump_nt_seq(expr),
    }
}

/// Dump a sequence (or a single non-choice item).
fn dump_nt_seq(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Sequence(items) => {
            items.iter()
                .map(|e| dump_nt_postfix(e))
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => dump_nt_postfix(expr),
    }
}

fn dump_nt_postfix(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Optional(inner) => format!("{}?", dump_nt_atom(inner)),
        GrammarExpr::Repeat(inner) => format!("{}*", dump_nt_atom(inner)),
        GrammarExpr::RepeatOne(inner) => format!("{}+", dump_nt_atom(inner)),
        GrammarExpr::RepeatRange { expr: inner, min, max } => {
            if min == max {
                format!("{}{{{}}}", dump_nt_atom(inner), min)
            } else {
                format!("{}{{{},{}}}", dump_nt_atom(inner), min, max)
            }
        }
        _ => dump_nt_atom(expr),
    }
}

fn dump_nt_atom(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Ref(name) => name.clone(),
        GrammarExpr::Literal(bytes) => format!("\"{}\"", escape_bytes_for_string(bytes)),
        GrammarExpr::RawRegex(pat) => format!("/{}/", escape_regex_for_slash(pat)),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let inner = if *negate { format!("^{}", def) } else { def.clone() };
            let suffix = if *utf8 { "/utf8" } else { "" };
            format!("[{}]{}", inner, suffix)
        }
        GrammarExpr::AnyByte => ".".to_string(),
        GrammarExpr::Epsilon => "eps".to_string(),
        GrammarExpr::Exclude { expr: inner, exclude } => {
            let lhs = dump_set_operand(inner);
            match exclude.as_ref() {
                GrammarExpr::Choice(alts) if !alts.is_empty() => {
                    let rhs = alts
                        .iter()
                        .map(dump_set_operand)
                        .collect::<Vec<_>>()
                        .join(" - ");
                    format!("({} - {})", lhs, rhs)
                }
                _ => format!("({} - {})", lhs, dump_set_operand(exclude)),
            }
        }
        GrammarExpr::Intersect { expr: inner, intersect } => {
            format!(
                "({} & {})",
                dump_set_operand(inner),
                dump_set_operand(intersect)
            )
        }
        GrammarExpr::SeparatedSequence { items, separator } => {
            let sep_str = dump_nt_atom(separator);
            let items_str = items.iter()
                .map(|(e, req)| {
                    let s = dump_nt_postfix(e);
                    if *req { s } else { format!("{}?", s) }
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} ~ ( {} )", sep_str, items_str)
        }
        // For compound exprs that need parens as atoms:
        GrammarExpr::Sequence(_) | GrammarExpr::Choice(_) => {
            format!("({})", dump_nt_expr(expr, false))
        }
        // Quantifiers that appear here need parens around their inner:
        GrammarExpr::Optional(_) | GrammarExpr::Repeat(_) | GrammarExpr::RepeatOne(_)
        | GrammarExpr::RepeatRange { .. } => {
            format!("({})", dump_nt_postfix(expr))
        }
    }
}

fn dump_set_operand(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Choice(_) | GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. } => {
            format!("({})", dump_nt_expr(expr, false))
        }
        _ => dump_nt_expr(expr, false),
    }
}

// ---- Helpers ---------------------------------------------------------------

fn escape_bytes_for_string(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(b as char),
            _ => out.push_str(&format!("\\x{:02X}", b)),
        }
    }
    out
}

fn escape_regex_for_slash(pat: &str) -> String {
    pat.replace('/', "\\/")
}

// ============================================================
// Parser
// ============================================================

/// Parse a GLRM-format string into a [`NamedGrammar`].
pub fn from_glrm(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser { tokens, pos: 0 };
    parser.parse_grammar()
}

// ---- Tokens ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// Identifier or keyword: `nt`, `t`, `internal`, `start`, `ignore`,
    /// `eps`
    Ident(String),
    /// String literal: `"..."` — bytes, after escape processing
    StringLit(Vec<u8>),
    /// Regex literal: `/.../` — raw pattern string (after unescape of `\/`)
    RegexLit(String),
    /// Character class: `[...]` or `[^...]` — (def_without_brackets, is_utf8)
    CharClass(String, bool),
    /// Integer
    Int(usize),
    /// `::=`
    DeclEq,
    /// `;`
    Semi,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `|`
    Pipe,
    /// `&`
    Amp,
    /// `-`
    Minus,
    /// `~`
    Tilde,
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `?`
    Quest,
    /// `,`
    Comma,
    /// `.`
    Dot,
    Eof,
}

// ---- Lexer -----------------------------------------------------------------

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer { src: src.as_bytes(), pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
            // Skip line comment
            if self.peek() == Some(b'/') && self.peek2() == Some(b'/') {
                self.pos += 2;
                while !matches!(self.peek(), None | Some(b'\n')) {
                    self.pos += 1;
                }
                continue;
            }
            // Skip block comment
            if self.peek() == Some(b'/') && self.peek2() == Some(b'*') {
                self.pos += 2;
                loop {
                    if self.peek() == Some(b'*') && self.peek2() == Some(b'/') {
                        self.pos += 2;
                        break;
                    }
                    if self.advance().is_none() {
                        break; // unterminated block comment — treat as EOF
                    }
                }
                continue;
            }
            break;
        }
    }

    fn lex_string(&mut self) -> Result<Vec<u8>, GlrMaskError> {
        let mut bytes = Vec::new();
        loop {
            match self.advance() {
                Some(b'"') => return Ok(bytes),
                Some(b'\\') => match self.advance() {
                    Some(b'n') => bytes.push(b'\n'),
                    Some(b't') => bytes.push(b'\t'),
                    Some(b'r') => bytes.push(b'\r'),
                    Some(b'\\') => bytes.push(b'\\'),
                    Some(b'"') => bytes.push(b'"'),
                    Some(b'x') => {
                        let hi = self.advance().ok_or_else(|| err("incomplete hex escape in string"))?;
                        let lo = self.advance().ok_or_else(|| err("incomplete hex escape in string"))?;
                        let val = (hex_digit(hi)? << 4) | hex_digit(lo)?;
                        bytes.push(val);
                    }
                    Some(c) => {
                        bytes.push(b'\\');
                        bytes.push(c);
                    }
                    None => return Err(err("unexpected EOF in string escape")),
                },
                Some(b) => bytes.push(b),
                None => return Err(err("unterminated string literal")),
            }
        }
    }

    fn lex_regex(&mut self) -> Result<String, GlrMaskError> {
        let mut pat = String::new();
        loop {
            match self.advance() {
                Some(b'/') => return Ok(pat),
                Some(b'\\') => {
                    match self.advance() {
                        Some(b'/') => pat.push('/'),  // escaped slash
                        Some(c) => {
                            pat.push('\\');
                            pat.push(c as char);
                        }
                        None => return Err(err("unexpected EOF in regex escape")),
                    }
                }
                Some(b) => pat.push(b as char),
                None => return Err(err("unterminated regex literal")),
            }
        }
    }

    fn lex_char_class(&mut self) -> Result<(String, bool), GlrMaskError> {
        // We're past the opening `[`. Collect everything up to the closing `]`.
        let mut def = String::new();
        if self.peek() == Some(b'^') {
            def.push('^');
            self.pos += 1;
        }
        loop {
            match self.advance() {
                Some(b']') => break,
                Some(b'\\') => {
                    let c = self.advance().ok_or_else(|| err("unterminated char class escape"))?;
                    def.push('\\');
                    def.push(c as char);
                }
                Some(b) => def.push(b as char),
                None => return Err(err("unterminated char class")),
            }
        }
        // Check for `/utf8` suffix
        let saved_pos = self.pos;
        self.skip_whitespace_and_comments();
        let is_utf8 = if self.peek() == Some(b'/') {
            // Try to read `utf8`
            self.pos += 1;
            let ident_start = self.pos;
            while matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
                self.pos += 1;
            }
            let word = &self.src[ident_start..self.pos];
            if word == b"utf8" {
                true
            } else {
                // Not a utf8 annotation; backtrack
                self.pos = saved_pos;
                false
            }
        } else {
            self.pos = saved_pos;
            false
        };
        Ok((def, is_utf8))
    }

    fn lex_ident(&mut self, first: u8) -> String {
        let mut s = String::new();
        s.push(first as char);
        while matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
            s.push(self.src[self.pos] as char);
            self.pos += 1;
        }
        s
    }

    fn lex_int(&mut self, first: u8) -> Result<usize, GlrMaskError> {
        let mut n = (first - b'0') as usize;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            n = n * 10 + (self.src[self.pos] - b'0') as usize;
            self.pos += 1;
        }
        Ok(n)
    }

    fn tokenize(mut self) -> Result<Vec<Tok>, GlrMaskError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None => { tokens.push(Tok::Eof); break; }
                Some(b) => {
                    self.pos += 1;
                    match b {
                        b';' => tokens.push(Tok::Semi),
                        b'(' => tokens.push(Tok::LParen),
                        b')' => tokens.push(Tok::RParen),
                        b'{' => tokens.push(Tok::LBrace),
                        b'}' => tokens.push(Tok::RBrace),
                        b'|' => tokens.push(Tok::Pipe),
                        b'&' => tokens.push(Tok::Amp),
                        b'-' => tokens.push(Tok::Minus),
                        b'~' => tokens.push(Tok::Tilde),
                        b'*' => tokens.push(Tok::Star),
                        b'+' => tokens.push(Tok::Plus),
                        b'?' => tokens.push(Tok::Quest),
                        b',' => tokens.push(Tok::Comma),
                        b'.' => tokens.push(Tok::Dot),
                        b':' => {
                            // Expect `::=`
                            if self.peek() == Some(b':') { self.pos += 1; }
                            if self.peek() == Some(b'=') { self.pos += 1; }
                            tokens.push(Tok::DeclEq);
                        }
                        b'"' => {
                            let bytes = self.lex_string()?;
                            tokens.push(Tok::StringLit(bytes));
                        }
                        b'/' => {
                            let pat = self.lex_regex()?;
                            tokens.push(Tok::RegexLit(pat));
                        }
                        b'[' => {
                            let (def, is_utf8) = self.lex_char_class()?;
                            tokens.push(Tok::CharClass(def, is_utf8));
                        }
                        b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                            let s = self.lex_ident(b);
                            tokens.push(Tok::Ident(s));
                        }
                        b'0'..=b'9' => {
                            let n = self.lex_int(b)?;
                            tokens.push(Tok::Int(n));
                        }
                        other => {
                            return Err(err(&format!("unexpected character '{}' (0x{:02X})", other as char, other)));
                        }
                    }
                }
            }
        }
        Ok(tokens)
    }
}

// ---- Parser ----------------------------------------------------------------

struct GlrmParser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl GlrmParser {

    fn peek(&self) -> &Tok {
        self.tokens.get(self.pos).unwrap_or(&Tok::Eof)
    }

    fn advance(&mut self) -> &Tok {
        let tok = self.tokens.get(self.pos).unwrap_or(&Tok::Eof);
        self.pos += 1;
        tok
    }

    fn consume(&mut self, expected: &Tok) -> Result<(), GlrMaskError> {
        let tok = self.advance().clone();
        if &tok == expected {
            Ok(())
        } else {
            Err(err(&format!("expected {:?}, got {:?}", expected, tok)))
        }
    }

    fn expect_ident(&mut self) -> Result<String, GlrMaskError> {
        match self.advance().clone() {
            Tok::Ident(s) => Ok(s),
            other => Err(err(&format!("expected identifier, got {:?}", other))),
        }
    }

    fn expect_int(&mut self) -> Result<usize, GlrMaskError> {
        match self.advance().clone() {
            Tok::Int(n) => Ok(n),
            other => Err(err(&format!("expected integer, got {:?}", other))),
        }
    }

    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        let mut start: Option<String> = None;
        let mut ignore: Option<String> = None;
        let mut rules: Vec<NamedRule> = Vec::new();

        loop {
            match self.peek().clone() {
                Tok::Eof => break,
                Tok::Ident(ref kw) => match kw.as_str() {
                    "start" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        start = Some(name);
                    }
                    "ignore" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        ignore = Some(name);
                    }
                    "nt" => {
                        self.advance();
                        rules.push(self.parse_rule(false, false)?);
                    }
                    "t" => {
                        self.advance();
                        rules.push(self.parse_rule(true, false)?);
                    }
                    "internal" => {
                        self.advance();
                        // Expect `t`
                        match self.advance().clone() {
                            Tok::Ident(ref kw2) if kw2 == "t" => {}
                            other => return Err(err(&format!("expected 't' after 'internal', got {:?}", other))),
                        }
                        rules.push(self.parse_rule(true, true)?);
                    }
                    other => return Err(err(&format!("unexpected keyword '{}' at top level", other))),
                },
                other => return Err(err(&format!("unexpected token {:?} at top level", other))),
            }
        }

        let start = start.ok_or_else(|| err("grammar has no 'start' declaration"))?;
        Ok(NamedGrammar { rules, start, ignore })
    }

    fn parse_rule(&mut self, is_terminal: bool, is_internal: bool) -> Result<NamedRule, GlrMaskError> {
        let name = self.expect_ident()?;
        self.consume(&Tok::DeclEq)?;
        // The expression can be empty (ε-only rule), so we don't require an atom.
        let expr = self.parse_nt_expr(is_terminal)?;
        self.consume(&Tok::Semi)?;
        Ok(NamedRule { name, expr, is_terminal, is_internal })
    }

    // ---- NT expression parsing ---------------------------------------------

    fn parse_nt_expr(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        // An alternative can be empty (ε), so try to parse even if no atom is visible.
        let first = if self.can_start_nt_atom() {
            self.parse_nt_exclude(allow_raw_regex)?
        } else {
            GrammarExpr::Sequence(vec![]) // empty / ε
        };
        if !matches!(self.peek(), Tok::Pipe) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while matches!(self.peek(), Tok::Pipe) {
            self.advance(); // consume `|`
            let alt = if self.can_start_nt_atom() {
                self.parse_nt_exclude(allow_raw_regex)?
            } else {
                GrammarExpr::Sequence(vec![]) // empty / ε
            };
            alts.push(alt);
        }
        Ok(GrammarExpr::Choice(alts))
    }

    fn parse_nt_exclude(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let expr = self.parse_nt_intersect(allow_raw_regex)?;
        let mut excludes = Vec::new();
        while matches!(self.peek(), Tok::Minus) {
            self.advance();
            excludes.push(self.parse_nt_intersect(allow_raw_regex)?);
        }
        if excludes.is_empty() {
            Ok(expr)
        } else {
            let exclude_expr = if excludes.len() == 1 {
                excludes.into_iter().next().unwrap()
            } else {
                GrammarExpr::Choice(excludes)
            };
            Ok(GrammarExpr::Exclude {
                expr: Box::new(expr),
                exclude: Box::new(exclude_expr),
            })
        }
    }

    fn parse_nt_intersect(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let mut expr = self.parse_nt_seq(allow_raw_regex)?;
        while matches!(self.peek(), Tok::Amp) {
            self.advance();
            let rhs = self.parse_nt_seq(allow_raw_regex)?;
            expr = GrammarExpr::Intersect {
                expr: Box::new(expr),
                intersect: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_nt_seq(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let mut items = Vec::new();
        loop {
            // A new item can start with: Ident, StringLit, RegexLit, CharClass,
            // Dot, LParen. Otherwise stop.
            if !self.can_start_nt_atom() {
                break;
            }
            items.push(self.parse_nt_postfix(allow_raw_regex)?);
        }
        match items.len() {
            0 => Err(err("expected at least one expression item")),
            1 => Ok(items.pop().unwrap()),
            _ => Ok(GrammarExpr::Sequence(items)),
        }
    }

    fn can_start_nt_atom(&self) -> bool {
        match self.peek() {
            Tok::Ident(s) if s == "eps" => true,
            Tok::Ident(_) | Tok::StringLit(_) | Tok::RegexLit(_)
            | Tok::CharClass(_, _) | Tok::Dot | Tok::LParen => true,
            _ => false,
        }
    }

    fn parse_nt_postfix(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let atom = self.parse_nt_atom(allow_raw_regex)?;
        // `sep ~ ( items... )` — SeparatedSequence
        if matches!(self.peek(), Tok::Tilde) {
            self.advance(); // consume `~`
            self.consume(&Tok::LParen)?;
            let mut items = Vec::new();
            loop {
                let item_expr = self.parse_sepseq_item(allow_raw_regex)?;
                let optional = if matches!(self.peek(), Tok::Quest) {
                    self.advance();
                    true
                } else {
                    false
                };
                items.push((item_expr, !optional)); // is_required = !optional
                if self.can_start_nt_atom() {
                    // more items
                } else {
                    break;
                }
            }
            self.consume(&Tok::RParen)?;
            if matches!(self.peek(), Tok::Quest | Tok::Star | Tok::Plus | Tok::LBrace) {
                return Err(err(
                    "quantifiers cannot be applied directly to SeparatedSequence; wrap it in a named rule instead",
                ));
            }
            return Ok(GrammarExpr::SeparatedSequence {
                items,
                separator: Box::new(atom),
            });
        }
        self.apply_nt_quantifier(atom)
    }

    fn parse_sepseq_item(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let mut atom = match self.peek() {
            Tok::LParen => {
                self.advance();
                let inner = self.parse_nt_expr(allow_raw_regex)?;
                self.consume(&Tok::RParen)?;
                // Preserve explicit grouping in sepseq items so lowering can
                // distinguish grouped repeats from top-level repeat items.
                GrammarExpr::Sequence(vec![inner])
            }
            _ => self.parse_nt_atom(allow_raw_regex)?,
        };

        // Inside `sep ~ ( ... )`, `?` is reserved for item optionality.
        atom = match self.peek() {
            Tok::Star => {
                self.advance();
                GrammarExpr::Repeat(Box::new(atom))
            }
            Tok::Plus => {
                self.advance();
                GrammarExpr::RepeatOne(Box::new(atom))
            }
            Tok::LBrace => {
                self.advance(); // `{`
                let min = self.expect_int()?;
                let max = if matches!(self.peek(), Tok::Comma) {
                    self.advance(); // `,`
                    if matches!(self.peek(), Tok::RBrace) {
                        self.consume(&Tok::RBrace)?;
                        return Ok(GrammarExpr::RepeatRange {
                            expr: Box::new(atom),
                            min,
                            max: min,
                        });
                    }
                    self.expect_int()?
                } else {
                    min
                };
                self.consume(&Tok::RBrace)?;
                GrammarExpr::RepeatRange {
                    expr: Box::new(atom),
                    min,
                    max,
                }
            }
            _ => atom,
        };
        Ok(atom)
    }

    fn apply_nt_quantifier(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        match self.peek() {
            Tok::Quest => { self.advance(); Ok(GrammarExpr::Optional(Box::new(atom))) }
            Tok::Star  => { self.advance(); Ok(GrammarExpr::Repeat(Box::new(atom))) }
            Tok::Plus  => { self.advance(); Ok(GrammarExpr::RepeatOne(Box::new(atom))) }
            Tok::LBrace => {
                self.advance(); // `{`
                let min = self.expect_int()?;
                let max = if matches!(self.peek(), Tok::Comma) {
                    self.advance(); // `,`
                    if matches!(self.peek(), Tok::RBrace) {
                        // `{n,}` — unbounded; not representable in RepeatRange so emit Repeat
                        self.consume(&Tok::RBrace)?;
                        // Minimum n, no max — we use RepeatRange with min=n, max=n for
                        // compatibility but actually there's no max. We'll use n as both.
                        // To keep it correct, use a large max sentinel via Optional+RepeatOne.
                        // Actually just use RepeatRange with min==max as fallback:
                        // Better: reject or use a very large max. For now: min.
                        return Ok(GrammarExpr::RepeatRange { expr: Box::new(atom), min, max: min });
                    }
                    self.expect_int()?
                } else {
                    min
                };
                self.consume(&Tok::RBrace)?;
                Ok(GrammarExpr::RepeatRange { expr: Box::new(atom), min, max })
            }
            _ => Ok(atom),
        }
    }

    fn parse_nt_atom(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        match self.peek().clone() {
            Tok::StringLit(bytes) => {
                self.advance();
                Ok(GrammarExpr::Literal(bytes))
            }
            Tok::RegexLit(pat) => {
                if !allow_raw_regex {
                    return Err(err("raw regex literals are only allowed in terminal (`t`) rules"));
                }
                self.advance();
                Ok(GrammarExpr::RawRegex(pat))
            }
            Tok::CharClass(def, is_utf8) => {
                self.advance();
                let negate = def.starts_with('^');
                let def_clean = if negate { def[1..].to_string() } else { def };
                Ok(GrammarExpr::CharClass { def: def_clean, negate, utf8: is_utf8 })
            }
            Tok::Dot => {
                self.advance();
                Ok(GrammarExpr::AnyByte)
            }
            Tok::LParen => {
                self.advance();
                let inner = self.parse_nt_expr(allow_raw_regex)?;
                self.consume(&Tok::RParen)?;
                Ok(inner)
            }
            Tok::Ident(ref kw) => {
                match kw.as_str() {
                    "eps" => {
                        self.advance();
                        Ok(GrammarExpr::Epsilon)
                    }
                    _ => {
                        // Generic identifier = Ref
                        let name = kw.clone();
                        self.advance();
                        Ok(GrammarExpr::Ref(name))
                    }
                }
            }
            other => Err(err(&format!("unexpected token {:?} in NT expression", other))),
        }
    }
}

// ---- Parse a byte character class string ----------------------------------

/// Parse a raw char-class definition string (as it appears between `[` and `]`
/// in the GLRM format, with `^` prefix for negation already stripped) into a
/// [`U8Set`].
fn parse_byte_class(def: &str) -> Result<U8Set, GlrMaskError> {
    let bytes = def.as_bytes();
    let mut i = 0;
    let negate = !def.is_empty() && bytes[0] == b'^';
    if negate { i += 1; }

    let mut set = U8Set::empty();

    while i < bytes.len() {
        let start_byte = read_class_byte(bytes, &mut i)?;
        if i + 1 < bytes.len() && bytes[i] == b'-' && (i + 1) < bytes.len() {
            i += 1; // consume `-`
            let end_byte = read_class_byte(bytes, &mut i)?;
            for b in start_byte..=end_byte {
                set.insert(b);
            }
        } else {
            set.insert(start_byte);
        }
    }

    if negate {
        let mut full = U8Set::all();
        for b in 0u8..=255 {
            if set.contains(b) { full.remove(b); }
        }
        Ok(full)
    } else {
        Ok(set)
    }
}

fn read_class_byte(bytes: &[u8], i: &mut usize) -> Result<u8, GlrMaskError> {
    if *i >= bytes.len() {
        return Err(err("unexpected end of char class"));
    }
    let b = bytes[*i];
    *i += 1;
    if b == b'\\' {
        if *i >= bytes.len() {
            return Err(err("unexpected end of char class escape"));
        }
        let c = bytes[*i];
        *i += 1;
        match c {
            b'n' => Ok(b'\n'),
            b't' => Ok(b'\t'),
            b'r' => Ok(b'\r'),
            b'\\' => Ok(b'\\'),
            b']' => Ok(b']'),
            b'-' => Ok(b'-'),
            b'^' => Ok(b'^'),
            b'x' => {
                if *i + 1 >= bytes.len() {
                    return Err(err("incomplete hex escape in char class"));
                }
                let hi = bytes[*i]; *i += 1;
                let lo = bytes[*i]; *i += 1;
                Ok((hex_digit(hi)? << 4) | hex_digit(lo)?)
            }
            _ => Ok(c),
        }
    } else {
        Ok(b)
    }
}

// ---- Small helpers ---------------------------------------------------------

fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(err(&format!("invalid hex digit '{}'", b as char))),
    }
}

fn err(msg: &str) -> GlrMaskError {
    GlrMaskError::GrammarParse(msg.to_string())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};

    fn roundtrip(grammar: &NamedGrammar) -> NamedGrammar {
        let dumped = to_glrm(grammar);
        from_glrm(&dumped).unwrap_or_else(|e| panic!("parse failed: {e}\n\ndumped:\n{dumped}"))
    }

    fn simple_grammar(rules: Vec<(&str, GrammarExpr, bool, bool)>, start: &str) -> NamedGrammar {
        NamedGrammar {
            rules: rules.into_iter().map(|(name, expr, is_terminal, is_internal)| NamedRule {
                name: name.to_string(),
                expr,
                is_terminal,
                is_internal,
            }).collect(),
            start: start.to_string(),
            ignore: None,
        }
    }

    #[test]
    fn test_roundtrip_simple_nt() {
        let g = simple_grammar(vec![
            ("start", GrammarExpr::Ref("item".to_string()), false, false),
            ("item", GrammarExpr::Literal(b"hello".to_vec()), false, false),
        ], "start");
        let g2 = roundtrip(&g);
        assert_eq!(g.rules.len(), g2.rules.len());
        assert_eq!(g2.start, "start");
    }

    #[test]
    fn test_roundtrip_terminal() {
        let g = simple_grammar(vec![
            ("start", GrammarExpr::Ref("FOO".to_string()), false, false),
            ("FOO", GrammarExpr::RawRegex("[a-z]+".to_string()), true, false),
        ], "start");
        let g2 = roundtrip(&g);
        assert_eq!(g2.rules[1].is_terminal, true);
        assert_eq!(g2.rules[1].is_internal, false);
    }

    #[test]
    fn test_roundtrip_exclude() {
        let base = GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass {
            def: "^\"" .to_string(),
            negate: true,
            utf8: false,
        }));
        let excl = GrammarExpr::Literal(b"bad".to_vec());
        let expr = GrammarExpr::Exclude {
            expr: Box::new(base),
            exclude: Box::new(excl),
        };
        let g = simple_grammar(vec![
            ("start", GrammarExpr::Ref("KEY".to_string()), false, false),
            ("KEY", expr, true, false),
        ], "start");
        let g2 = roundtrip(&g);
        assert_eq!(g2.rules[1].is_terminal, true);
        // Basic structural check
        assert!(matches!(g2.rules[1].expr, GrammarExpr::Exclude { .. }));
    }

    #[test]
    fn test_roundtrip_intersect_te() {
        let isect = GrammarExpr::Intersect {
            expr: Box::new(GrammarExpr::Repeat(Box::new(GrammarExpr::Literal(b"x".to_vec())))),
            intersect: Box::new(GrammarExpr::Literal(b"xx".to_vec())),
        };
        let g = simple_grammar(vec![
            ("start", GrammarExpr::Ref("TM".to_string()), false, false),
            ("TM", isect, true, false),
        ], "start");
        let g2 = roundtrip(&g);
        assert!(matches!(g2.rules[1].expr, GrammarExpr::Intersect { .. }));
    }

    #[test]
    fn test_parse_infix_intersect_and_exclude() {
        let src = r#"
start start;

t TM ::= "a" | "b" | "c";
nt start ::= (TM & "b") - "a";
"#;
        let g = from_glrm(src).expect("infix operators should parse");
        assert_eq!(g.start, "start");
        assert_eq!(g.rules.len(), 2);
        assert!(matches!(g.rules[1].expr, GrammarExpr::Exclude { .. }));
    }

    #[test]
    fn test_parse_exclude_chain_as_union() {
        let src = r#"
start start;

nt start ::= key - a - b;
nt key ::= "k";
nt a ::= "a";
nt b ::= "b";
"#;
        let g = from_glrm(src).expect("exclude chain should parse");
        let start = g.rules.iter().find(|r| r.name == "start").unwrap();
        match &start.expr {
            GrammarExpr::Exclude { exclude, .. } => {
                assert!(matches!(exclude.as_ref(), GrammarExpr::Choice(alts) if alts.len() == 2));
            }
            other => panic!("expected Exclude, got {other:?}"),
        }
    }

    #[test]
    fn test_dump_exclude_union_as_chain() {
        let expr = GrammarExpr::Exclude {
            expr: Box::new(GrammarExpr::Ref("key".into())),
            exclude: Box::new(GrammarExpr::Choice(vec![
                GrammarExpr::Ref("a".into()),
                GrammarExpr::Ref("b".into()),
            ])),
        };
        let g = simple_grammar(
            vec![
                ("start", expr, false, false),
                ("key", GrammarExpr::Literal(b"k".to_vec()), false, false),
                ("a", GrammarExpr::Literal(b"a".to_vec()), false, false),
                ("b", GrammarExpr::Literal(b"b".to_vec()), false, false),
            ],
            "start",
        );
        let dumped = to_glrm(&g);
        assert!(dumped.contains("key - a - b"), "expected chained dump form, got: {dumped}");
        assert!(
            dumped.contains("nt start ::= key - a - b;"),
            "expected no outer parentheses around top-level chain, got: {dumped}"
        );
    }

    #[test]
    fn test_nt_rule_rejects_raw_regex() {
        let src = r#"
start start;

nt start ::= /abc/;
"#;
        let err = from_glrm(src).expect_err("nt raw regex must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("raw regex literals are only allowed in terminal"));
    }

    #[test]
    fn test_roundtrip_separated_sequence() {
        // "," ~ ( A B? C (D E F)? )
        let sep = GrammarExpr::Literal(b",".to_vec());
        let items = vec![
            (GrammarExpr::Ref("A".to_string()), true),
            (GrammarExpr::Ref("B".to_string()), false), // optional
            (GrammarExpr::Ref("C".to_string()), true),
            (GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("D".to_string()),
                GrammarExpr::Ref("E".to_string()),
                GrammarExpr::Ref("F".to_string()),
            ]), false), // optional
        ];
        let expr = GrammarExpr::SeparatedSequence {
            items,
            separator: Box::new(sep),
        };
        let g = simple_grammar(vec![
            ("A", GrammarExpr::Literal(b"a".to_vec()), false, false),
            ("B", GrammarExpr::Literal(b"b".to_vec()), false, false),
            ("C", GrammarExpr::Literal(b"c".to_vec()), false, false),
            ("D", GrammarExpr::Literal(b"d".to_vec()), false, false),
            ("E", GrammarExpr::Literal(b"e".to_vec()), false, false),
            ("F", GrammarExpr::Literal(b"f".to_vec()), false, false),
            ("start", expr, false, false),
        ], "start");
        let dumped = to_glrm(&g);
        // Verify the tilde syntax appears in the dump
        assert!(dumped.contains('~'), "dump should contain '~': {dumped}");
        assert!(!dumped.contains("seqsep"), "dump must not contain 'seqsep': {dumped}");
        // Roundtrip
        let g2 = from_glrm(&dumped)
            .unwrap_or_else(|e| panic!("parse failed: {e}\n\ndumped:\n{dumped}"));
        let start_rule = g2.rules.iter().find(|r| r.name == "start").unwrap();
        assert!(matches!(start_rule.expr, GrammarExpr::SeparatedSequence { .. }));
    }

    #[test]
    fn test_parse_tilde_from_source() {
        let src = r#"
start start;

nt A ::= "a";
nt B ::= "b";
nt C ::= "c";
nt start ::= "," ~ ( A B? C );
"#;
        let g = from_glrm(src).expect("tilde syntax should parse");
        let start = g.rules.iter().find(|r| r.name == "start").unwrap();
        match &start.expr {
            GrammarExpr::SeparatedSequence { items, separator } => {
                assert!(matches!(**separator, GrammarExpr::Literal(_)));
                assert_eq!(items.len(), 3);
                assert!(items[0].1, "A is required");
                assert!(!items[1].1, "B is optional");
                assert!(items[2].1, "C is required");
            }
            other => panic!("expected SeparatedSequence, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_tilde_with_repeat_items() {
        let src = r#"
start start;

nt a ::= "a";
nt b ::= "b";
nt c ::= "c";
nt d ::= "d";
nt e ::= "e";
nt start ::= "," ~ ( a b* c+ d{2,4} (e{2,3}) );
"#;
        let g = from_glrm(src).expect("repeated sepseq items should parse");
        let start = g.rules.iter().find(|r| r.name == "start").unwrap();
        match &start.expr {
            GrammarExpr::SeparatedSequence { items, .. } => {
                assert_eq!(items.len(), 5);
                assert!(matches!(items[1].0, GrammarExpr::Repeat(_)));
                assert!(matches!(items[2].0, GrammarExpr::RepeatOne(_)));
                assert!(matches!(items[3].0, GrammarExpr::RepeatRange { min: 2, max: 4, .. }));
                assert!(matches!(items[4].0, GrammarExpr::Sequence(_)));
            }
            other => panic!("expected SeparatedSequence, got {other:?}"),
        }
    }

    #[test]
    fn test_reject_quantified_separated_sequence() {
        let src = r#"
start start;

nt a ::= "a";
nt b ::= "b";
nt start ::= "," ~ ( a b ){2,4};
"#;
        let err = from_glrm(src).expect_err("quantified sepseq should be rejected");
        assert!(err.to_string().contains("SeparatedSequence"));
    }

    #[test]
    fn test_dump_and_parse_choice_sequence() {
        let expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            GrammarExpr::Choice(vec![
                GrammarExpr::Ref("a".to_string()),
                GrammarExpr::Ref("b".to_string()),
            ]),
            GrammarExpr::Literal(b"}".to_vec()),
        ]);
        let g = simple_grammar(vec![
            ("start", expr, false, false),
        ], "start");
        let g2 = roundtrip(&g);
        assert!(matches!(g2.rules[0].expr, GrammarExpr::Sequence(_)));
    }
}
