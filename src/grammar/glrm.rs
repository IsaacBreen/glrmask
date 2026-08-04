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
//! [g <name> ::= { <grammar-body> };]
//! [lexer group <partition-name> ::= <TM-NAME> | "literal" | @literals | *, ...;]
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
//! A `g` declaration defines a named subgrammar. Its body is a complete GLRM
//! grammar scope with its own `start`, optional `ignore`, terminals, rules,
//! lexer groups, and nested subgrammars. Definitions are strictly scope-local:
//! a subgrammar cannot see its parent's definitions, and its private definitions
//! are not visible to the parent. Only the declared subgrammar name is visible
//! in the enclosing scope, where it is referenced like a nonterminal.
//!
//! Ignore is also scope-local. `ignore I;` admits `I*` before the first lexical
//! atom in that grammar scope, between lexical atoms, and after the last lexical
//! atom. It never splits a terminal match. A subgrammar with no `ignore`
//! declaration inherits nothing from its parent.
//!
//! ## Expressions (used for both NT and terminal rule bodies)
//!
//! | Syntax                          | Meaning                              |
//! |--------------------------------|--------------------------------------|
//! | `name`                         | Reference to a rule                  |
//! | `"text"`                       | Literal bytes                        |
//! | `/regex/`                      | Raw regex pattern (terminal rules only) |
//! | `[class]`, `[^class]`          | Byte character class                 |
//! | `[class]/utf8`                 | UTF-8 character class                |
//! | `.`                            | Any byte                             |
//! | `eps`                          | Epsilon (empty string)               |
//! | `@token(123)`                  | Exact LLM token id 123               |
//! | `a b c`                        | Sequence                             |
//! | `a \| b \| c`                  | Choice                               |
//! | `e?`, `e*`, `e+`              | Optional / Repeat / RepeatOne        |
//! | `e{n}`, `e{n,m}`              | RepeatRange                          |
//! | `(e)`                          | Grouping                             |
//! | `a - b`                        | GrammarExpr::Exclude                 |
//! | `a & b`                        | GrammarExpr::Intersect               |
//! | `sep ~ ( i1? i2 i3? )`           | SeparatedSequence                    |

use crate::GlrMaskError;
use crate::automata::unweighted_u32::dfa::Label;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::grammar::ast::{
    GrammarExpr, NamedGrammar, NamedRule, Quantifier, resolved_named_terminal_exprs,
};
use crate::grammar::expr_nfa::ExprNFA;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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
    let anonymous_literals = grammar.emitted_anonymous_literals();
    let literal_selector_partition = if !anonymous_literals.is_empty()
        && anonymous_literals
            .iter()
            .all(|literal| grammar.lexer_literal_partitions.contains_key(literal))
    {
        let mut counts = BTreeMap::<&str, usize>::new();
        for literal in &anonymous_literals {
            let partition = grammar
                .lexer_literal_partitions
                .get(literal)
                .expect("all anonymous literals checked above");
            *counts.entry(partition.as_str()).or_default() += 1;
        }
        counts
            .into_iter()
            .max_by(|(left_name, left_count), (right_name, right_count)| {
                left_name
                    .contains("literal")
                    .cmp(&right_name.contains("literal"))
                    .then_with(|| left_count.cmp(right_count))
                    .then_with(|| right_name.cmp(left_name))
            })
            .map(|(partition, _)| partition)
    } else {
        None
    };

    let mut lexer_groups = BTreeMap::<&str, Vec<String>>::new();
    for (terminal, partition) in &grammar.lexer_partitions {
        lexer_groups
            .entry(partition.as_str())
            .or_default()
            .push(terminal.clone());
    }
    for (literal, partition) in &grammar.lexer_literal_partitions {
        if literal_selector_partition == Some(partition.as_str())
            && anonymous_literals.contains(literal)
        {
            continue;
        }
        lexer_groups
            .entry(partition.as_str())
            .or_default()
            .push(format!("\"{}\"", escape_bytes_for_string(literal)));
    }
    if let Some(partition) = literal_selector_partition {
        lexer_groups
            .entry(partition)
            .or_default()
            .push("@literals".to_string());
    }
    if let Some(partition) = grammar.default_lexer_partition.as_deref() {
        lexer_groups
            .entry(partition)
            .or_default()
            .push("*".to_string());
    }
    for (partition, mut members) in lexer_groups {
        members.sort_unstable();
        out.push_str(&format!(
            "lexer group {} ::= {};\n",
            partition,
            members.join(", "),
        ));
    }
    out.push('\n');

    for rule in &grammar.rules {
        if !rule.is_terminal {
            if let GrammarExpr::ExprNFA(expr_nfa) = &rule.expr {
                out.push_str(&format!("fa {} ::= {{\n", rule.name));
                out.push_str(&dump_expr_nfa(expr_nfa));
                out.push_str("};\n");
                continue;
            }
        }
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

fn dump_expr_nfa(expr_nfa: &ExprNFA) -> String {
    let mut out = String::new();
    let starts = expr_nfa
        .nfa
        .start_states
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("  start {};\n", starts));

    let accepts = expr_nfa
        .nfa
        .states
        .iter()
        .enumerate()
        .filter_map(|(state_id, state)| state.is_accepting.then(|| state_id.to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("  accept {};\n\n", accepts));

    for (state_id, state) in expr_nfa.nfa.states.iter().enumerate() {
        for &target in &state.epsilons {
            out.push_str(&format!("  {state_id} --> {target};\n"));
        }
        for (&label, targets) in &state.transitions {
            let symbol = expr_nfa
                .symbol_for_label(label)
                .map(|expr| dump_nt_expr(expr, false))
                .unwrap_or_else(|| format!("/*invalid-symbol-{label}*/ eps"));
            for &target in targets {
                out.push_str(&format!("  {state_id} -- {symbol} --> {target};\n"));
            }
        }
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

fn dump_quantifier(quantifier: &Quantifier) -> String {
    match quantifier {
        Quantifier::Optional => "?".to_string(),
        Quantifier::ZeroPlus => "*".to_string(),
        Quantifier::OnePlus => "+".to_string(),
        Quantifier::Range(min, Some(max)) if min == max => format!("{{{}}}", min),
        Quantifier::Range(min, Some(max)) => format!("{{{},{}}}", min, max),
        Quantifier::Range(min, None) => format!("{{{},}}", min),
    }
}

fn dump_nt_postfix(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Quantified(inner, Quantifier::Optional) => format!("{}?", dump_nt_atom(inner)),
        GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => format!("{}*", dump_nt_atom(inner)),
        GrammarExpr::Quantified(inner, Quantifier::OnePlus) => format!("{}+", dump_nt_atom(inner)),
        GrammarExpr::Quantified(inner, Quantifier::Range(min, max)) => match max {
            Some(max) if min == max => format!("{}{{{}}}", dump_nt_atom(inner), min),
            Some(max) => format!("{}{{{},{}}}", dump_nt_atom(inner), min, max),
            None => format!("{}{{{},}}", dump_nt_atom(inner), min),
        },
        _ => dump_nt_atom(expr),
    }
}

fn dump_nt_atom(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Ref(name) => name.clone(),
        GrammarExpr::Grouped(inner) => format!("({})", dump_nt_expr(inner, false)),
        GrammarExpr::Literal(bytes) => format!("\"{}\"", escape_bytes_for_string(bytes)),
        GrammarExpr::SpecialToken(token_id) => format!("@token({token_id})"),
        GrammarExpr::RawRegex(pat) => format!("/{}/", escape_regex_for_slash(pat)),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let inner = if *negate { format!("^{}", def) } else { def.clone() };
            let suffix = if *utf8 { "/utf8" } else { "" };
            format!("[{}]{}", inner, suffix)
        }
        GrammarExpr::LexerDfa(_) => "LexerDfa".to_string(),
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
        GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
            let sep_str = dump_nt_atom(separator);
            let items_str = items.iter()
                .map(|(e, quantifier)| {
                    let mut s = dump_nt_atom(e);
                    if let Some(quantifier) = quantifier {
                        s.push_str(&dump_quantifier(quantifier));
                    }
                    s
                })
                .collect::<Vec<_>>()
                .join(" ");
            if *allow_empty {
                format!("{} ~ ( {} )", sep_str, items_str)
            } else {
                format!("{} ~+ ( {} )", sep_str, items_str)
            }
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            format!(
                "ExprNFA(states={}, symbols={})",
                expr_nfa.nfa.states.len(),
                expr_nfa.symbols.len()
            )
        }
        // For compound exprs that need parens as atoms:
        GrammarExpr::Sequence(_) | GrammarExpr::Choice(_) => {
            format!("({})", dump_nt_expr(expr, false))
        }
        // Quantifiers that appear here need parens around their inner:
        GrammarExpr::Quantified(_, Quantifier::Optional) | GrammarExpr::Quantified(_, Quantifier::ZeroPlus) | GrammarExpr::Quantified(_, Quantifier::OnePlus)
        | GrammarExpr::Quantified(_, Quantifier::Range(_, _)) => {
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
    /// `--`
    Dashes,
    /// `-->`
    Arrow,
    /// `~`
    Tilde,
    /// `*`
    Star,
    /// `@`
    At,
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

    fn lex_string(&mut self, delim: u8) -> Result<Vec<u8>, GlrMaskError> {
        let mut bytes = Vec::new();
        loop {
            match self.advance() {
                Some(b) if b == delim => return Ok(bytes),
                Some(b'\\') => match self.advance() {
                    Some(b'n') => bytes.push(b'\n'),
                    Some(b't') => bytes.push(b'\t'),
                    Some(b'r') => bytes.push(b'\r'),
                    Some(b'\\') => bytes.push(b'\\'),
                    Some(b'"') if delim == b'"' => bytes.push(b'"'),
                    Some(b'\'') if delim == b'\'' => bytes.push(b'\''),
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
                        b'-' => {
                            if self.peek() == Some(b'-') && self.peek2() == Some(b'>') {
                                self.pos += 2;
                                tokens.push(Tok::Arrow);
                            } else if self.peek() == Some(b'-') {
                                self.pos += 1;
                                tokens.push(Tok::Dashes);
                            } else {
                                tokens.push(Tok::Minus);
                            }
                        }
                        b'~' => tokens.push(Tok::Tilde),
                        b'*' => tokens.push(Tok::Star),
                        b'@' => tokens.push(Tok::At),
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
                        b'"' | b'\'' => {
                            let bytes = self.lex_string(b)?;
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

#[derive(Debug, Clone)]
struct ParsedSubgrammar {
    name: String,
    scope: ParsedGlrmScope,
}

#[derive(Debug, Clone)]
struct ParsedGlrmScope {
    rules: Vec<NamedRule>,
    subgrammars: Vec<ParsedSubgrammar>,
    start: String,
    ignore: Option<String>,
    lexer_partitions: BTreeMap<String, String>,
    lexer_literal_partitions: BTreeMap<Vec<u8>, String>,
    default_lexer_partition: Option<String>,
    all_literals_partition: Option<String>,
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
        let scope = self.parse_scope(false, "grammar")?;
        if scope.subgrammars.is_empty() {
            let grammar = named_grammar_for_scope(&scope)?;
            validate_ignore_terminal(&grammar, "grammar")?;
            return Ok(grammar);
        }
        flatten_scoped_grammar(scope)
    }

    fn parse_scope(
        &mut self,
        stop_at_rbrace: bool,
        scope_label: &str,
    ) -> Result<ParsedGlrmScope, GlrMaskError> {
        let mut start: Option<String> = None;
        let mut ignore: Option<String> = None;
        let mut rules: Vec<NamedRule> = Vec::new();
        let mut subgrammars = Vec::new();
        let mut lexer_partitions = BTreeMap::<String, String>::new();
        let mut lexer_literal_partitions = BTreeMap::<Vec<u8>, String>::new();
        let mut default_lexer_partition = None::<String>;
        let mut all_literals_partition = None::<String>;

        loop {
            match self.peek().clone() {
                Tok::Eof if stop_at_rbrace => {
                    return Err(err(&format!("unterminated {scope_label}")));
                }
                Tok::Eof => break,
                Tok::RBrace if stop_at_rbrace => {
                    self.advance();
                    break;
                }
                Tok::RBrace => {
                    return Err(err("unexpected '}' at top level"));
                }
                Tok::Ident(ref kw) => match kw.as_str() {
                    "start" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        if start.replace(name).is_some() {
                            return Err(err(&format!(
                                "{scope_label} has more than one 'start' declaration",
                            )));
                        }
                    }
                    "ignore" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        if ignore.replace(name).is_some() {
                            return Err(err(&format!(
                                "{scope_label} has more than one 'ignore' declaration",
                            )));
                        }
                    }
                    "g" | "subgrammar" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::DeclEq)?;
                        self.consume(&Tok::LBrace)?;
                        let child_label = format!("subgrammar '{name}'");
                        let scope = self.parse_scope(true, &child_label)?;
                        self.consume(&Tok::Semi)?;
                        subgrammars.push(ParsedSubgrammar { name, scope });
                    }
                    "lexer" => {
                        self.advance();
                        self.parse_lexer_group(
                            &mut lexer_partitions,
                            &mut lexer_literal_partitions,
                            &mut default_lexer_partition,
                            &mut all_literals_partition,
                        )?;
                    }
                    "nt" => {
                        self.advance();
                        rules.push(self.parse_rule(false, false)?);
                    }
                    "fa" | "nfa" => {
                        self.advance();
                        rules.push(self.parse_expr_nfa_rule()?);
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
                    other => {
                        return Err(err(&format!(
                            "unexpected keyword '{other}' in {scope_label}",
                        )));
                    }
                },
                other => {
                    return Err(err(&format!(
                        "unexpected token {:?} in {scope_label}",
                        other,
                    )));
                }
            }
        }

        let start = start.ok_or_else(|| err(&format!("{scope_label} has no 'start' declaration")))?;
        Ok(ParsedGlrmScope {
            rules,
            subgrammars,
            start,
            ignore,
            lexer_partitions,
            lexer_literal_partitions,
            default_lexer_partition,
            all_literals_partition,
        })
    }

    fn parse_lexer_group(
        &mut self,
        lexer_partitions: &mut BTreeMap<String, String>,
        lexer_literal_partitions: &mut BTreeMap<Vec<u8>, String>,
        default_lexer_partition: &mut Option<String>,
        all_literals_partition: &mut Option<String>,
    ) -> Result<(), GlrMaskError> {
        match self.advance().clone() {
            Tok::Ident(ref keyword) if keyword == "group" => {}
            other => {
                return Err(err(&format!(
                    "expected 'group' after 'lexer', got {:?}",
                    other,
                )));
            }
        }
        let partition = self.expect_ident()?;
        self.consume(&Tok::DeclEq)?;
        loop {
            match self.advance().clone() {
                Tok::Ident(terminal) => {
                    if lexer_partitions
                        .insert(terminal.clone(), partition.clone())
                        .is_some()
                    {
                        return Err(err(&format!(
                            "terminal '{terminal}' is assigned to more than one lexer group",
                        )));
                    }
                }
                Tok::StringLit(literal) => {
                    if lexer_literal_partitions
                        .insert(literal.clone(), partition.clone())
                        .is_some()
                    {
                        return Err(err(&format!(
                            "literal {:?} is assigned to more than one lexer group",
                            String::from_utf8_lossy(&literal),
                        )));
                    }
                }
                Tok::Star => {
                    if default_lexer_partition
                        .replace(partition.clone())
                        .is_some()
                    {
                        return Err(err(
                            "the catch-all '*' is assigned to more than one lexer group",
                        ));
                    }
                }
                Tok::At => match self.advance().clone() {
                    Tok::Ident(selector) if selector == "literals" => {
                        if all_literals_partition
                            .replace(partition.clone())
                            .is_some()
                        {
                            return Err(err(
                                "'@literals' is assigned to more than one lexer group",
                            ));
                        }
                    }
                    other => {
                        return Err(err(&format!(
                            "expected 'literals' after '@', got {:?}",
                            other,
                        )));
                    }
                },
                other => {
                    return Err(err(&format!(
                        "expected terminal name, literal, or '*', got {:?}",
                        other,
                    )));
                }
            }
            if matches!(self.peek(), Tok::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        self.consume(&Tok::Semi)
    }

    fn parse_rule(&mut self, is_terminal: bool, is_internal: bool) -> Result<NamedRule, GlrMaskError> {
        let name = self.expect_ident()?;
        self.consume(&Tok::DeclEq)?;
        // The expression can be empty (ε-only rule), so we don't require an atom.
        let expr = self.parse_nt_expr(is_terminal)?;
        self.consume(&Tok::Semi)?;
        Ok(NamedRule { name, expr, is_terminal, is_internal })
    }

    fn parse_expr_nfa_rule(&mut self) -> Result<NamedRule, GlrMaskError> {
        let name = self.expect_ident()?;
        self.consume(&Tok::DeclEq)?;
        self.consume(&Tok::LBrace)?;

        let mut nfa = NFA::new_empty();
        let mut symbols = Vec::<GrammarExpr>::new();
        let mut symbol_labels = HashMap::<GrammarExpr, Label>::new();

        loop {
            match self.peek().clone() {
                Tok::RBrace => {
                    self.advance();
                    break;
                }
                Tok::Ident(ref kw) if kw == "start" => {
                    self.advance();
                    nfa.start_states.clear();
                    loop {
                        let state = self.expect_int()? as u32;
                        ensure_nfa_state(&mut nfa, state);
                        nfa.start_states.push(state);
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                            continue;
                        }
                        break;
                    }
                    self.consume(&Tok::Semi)?;
                }
                Tok::Ident(ref kw) if kw == "accept" => {
                    self.advance();
                    if !matches!(self.peek(), Tok::Semi) {
                        loop {
                            let state = self.expect_int()? as u32;
                            ensure_nfa_state(&mut nfa, state);
                            nfa.set_accepting(state);
                            if matches!(self.peek(), Tok::Comma) {
                                self.advance();
                                continue;
                            }
                            break;
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                Tok::Int(_) => {
                    let from = self.expect_int()? as u32;
                    ensure_nfa_state(&mut nfa, from);
                    match self.peek() {
                        Tok::Arrow => {
                            self.advance();
                            let to = self.expect_int()? as u32;
                            ensure_nfa_state(&mut nfa, to);
                            nfa.add_epsilon(from, to);
                        }
                        Tok::Dashes => {
                            self.advance();
                            let symbol = self.parse_expr_nfa_transition_expr()?;
                            self.consume(&Tok::Arrow)?;
                            let to = self.expect_int()? as u32;
                            ensure_nfa_state(&mut nfa, to);
                            let label = intern_expr_nfa_symbol(
                                &mut symbols,
                                &mut symbol_labels,
                                symbol,
                            );
                            nfa.add_transition(from, label, to);
                        }
                        other => {
                            return Err(err(&format!(
                                "expected '--' or '-->' after FA transition source, got {:?}",
                                other
                            )));
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                other => return Err(err(&format!("unexpected token {:?} in FA definition", other))),
            }
        }

        self.consume(&Tok::Semi)?;
        if nfa.start_states.is_empty() {
            return Err(err("FA definition has no start state"));
        }
        Ok(NamedRule {
            name,
            expr: GrammarExpr::ExprNFA(Box::new(ExprNFA::new(nfa, symbols))),
            is_terminal: false,
            is_internal: false,
        })
    }

    fn parse_expr_nfa_transition_expr(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        if matches!(self.peek(), Tok::Arrow) {
            return Err(err(
                "FA transition expression cannot be empty; use epsilon transition syntax",
            ));
        }
        if !self.can_start_nt_atom() {
            return Err(err(&format!(
                "expected FA transition expression item before '-->', got {:?}",
                self.peek()
            )));
        }
        self.parse_nt_expr(false)
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
            excludes.push(self.parse_nt_exclude_rhs(allow_raw_regex)?);
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

    fn parse_nt_exclude_rhs(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        if matches!(self.peek(), Tok::LParen) {
            self.advance();
            let inner = self.parse_nt_expr(allow_raw_regex)?;
            self.consume(&Tok::RParen)?;
            return Ok(GrammarExpr::Grouped(Box::new(inner)));
        }

        let mut expr = self.parse_nt_postfix(allow_raw_regex)?;
        while matches!(self.peek(), Tok::Amp) {
            self.advance();
            let rhs = self.parse_nt_postfix(allow_raw_regex)?;
            expr = GrammarExpr::Intersect {
                expr: Box::new(expr),
                intersect: Box::new(rhs),
            };
        }
        if self.can_start_nt_atom() {
            return Err(err("invalid syntax: RHS sequence subtraction must be parenthesized"));
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
            | Tok::CharClass(_, _) | Tok::Dot | Tok::LParen | Tok::At => true,
            _ => false,
        }
    }

    fn parse_nt_postfix(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let atom = self.parse_nt_atom(allow_raw_regex)?;
        // `sep ~ ( items... )` / `sep ~+ ( items... )` — SeparatedSequence
        if matches!(self.peek(), Tok::Tilde) {
            self.advance(); // consume `~`
            let allow_empty = if matches!(self.peek(), Tok::Plus) {
                self.advance();
                false
            } else {
                true
            };
            self.consume(&Tok::LParen)?;
            let mut items = Vec::new();
            loop {
                let item = self.parse_sepseq_item(allow_raw_regex)?;
                items.push(item);
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
                allow_empty,
            });
        }
        self.apply_nt_quantifier(atom)
    }

    fn parse_sepseq_item(
        &mut self,
        allow_raw_regex: bool,
    ) -> Result<(GrammarExpr, Option<Quantifier>), GlrMaskError> {
        let atom = match self.peek() {
            Tok::LParen => {
                self.advance();
                let inner = self.parse_nt_expr(allow_raw_regex)?;
                self.consume(&Tok::RParen)?;
                GrammarExpr::Grouped(Box::new(inner))
            }
            _ => self.parse_nt_atom(allow_raw_regex)?,
        };

        let quantifier = match self.peek() {
            Tok::Quest => {
                self.advance();
                Some(Quantifier::Optional)
            }
            Tok::Star => {
                self.advance();
                Some(Quantifier::ZeroPlus)
            }
            Tok::Plus => {
                self.advance();
                Some(Quantifier::OnePlus)
            }
            Tok::LBrace => {
                self.advance();
                let min = self.expect_int()?;
                let max = if matches!(self.peek(), Tok::Comma) {
                    self.advance();
                    if matches!(self.peek(), Tok::RBrace) {
                        None
                    } else {
                        Some(self.expect_int()?)
                    }
                } else {
                    Some(min)
                };
                self.consume(&Tok::RBrace)?;
                Some(Quantifier::Range(min, max))
            }
            _ => None,
        };

        if quantifier.is_some() && matches!(self.peek(), Tok::Quest | Tok::Star | Tok::Plus | Tok::LBrace) {
            return Err(err(
                "only one postfix quantifier may bind to a separated-sequence item; use extra parentheses, e.g. `sep ~ ( (expr+)? )`, when the outer quantifier should bind to the separator",
            ));
        }

        Ok((atom, quantifier))
    }

    fn apply_nt_quantifier(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        match self.peek() {
            Tok::Quest => { self.advance(); Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Optional)) }
            Tok::Star  => { self.advance(); Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::ZeroPlus)) }
            Tok::Plus  => { self.advance(); Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::OnePlus)) }
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
                        return Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Range(min, None)));
                    }
                    self.expect_int()?
                } else {
                    min
                };
                self.consume(&Tok::RBrace)?;
                Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Range(min, Some(max))))
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
            Tok::At => {
                self.advance();
                match self.advance().clone() {
                    Tok::Ident(keyword) if keyword == "token" => {}
                    other => {
                        return Err(err(&format!(
                            "expected 'token' after '@' in expression, got {:?}",
                            other
                        )));
                    }
                }
                self.consume(&Tok::LParen)?;
                let token_id = self.expect_int()?;
                let token_id = u32::try_from(token_id)
                    .map_err(|_| err("special LLM token id does not fit in u32"))?;
                self.consume(&Tok::RParen)?;
                Ok(GrammarExpr::SpecialToken(token_id))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopedSymbolKind {
    Nonterminal,
    Terminal,
    InternalTerminal,
    Subgrammar,
}

fn named_grammar_for_scope(scope: &ParsedGlrmScope) -> Result<NamedGrammar, GlrMaskError> {
    let mut grammar = NamedGrammar {
        rules: scope.rules.clone(),
        start: scope.start.clone(),
        ignore: scope.ignore.clone(),
        lexer_partitions: scope.lexer_partitions.clone(),
        lexer_literal_partitions: scope.lexer_literal_partitions.clone(),
        default_lexer_partition: scope.default_lexer_partition.clone(),
    };
    if let Some(partition) = scope.all_literals_partition.as_deref() {
        for literal in grammar.emitted_anonymous_literals() {
            grammar
                .lexer_literal_partitions
                .entry(literal)
                .or_insert_with(|| partition.to_string());
        }
    }
    Ok(grammar)
}

fn scope_symbol_kinds(
    scope: &ParsedGlrmScope,
    scope_label: &str,
) -> Result<BTreeMap<String, ScopedSymbolKind>, GlrMaskError> {
    let mut symbols = BTreeMap::new();
    for rule in &scope.rules {
        let kind = match (rule.is_terminal, rule.is_internal) {
            (false, _) => ScopedSymbolKind::Nonterminal,
            (true, false) => ScopedSymbolKind::Terminal,
            (true, true) => ScopedSymbolKind::InternalTerminal,
        };
        if let Some(previous) = symbols.insert(rule.name.clone(), kind)
            && previous != kind
        {
            return Err(err(&format!(
                "definition '{}' has conflicting kinds in {scope_label}",
                rule.name,
            )));
        }
    }
    for subgrammar in &scope.subgrammars {
        if symbols
            .insert(subgrammar.name.clone(), ScopedSymbolKind::Subgrammar)
            .is_some()
        {
            return Err(err(&format!(
                "subgrammar '{}' conflicts with another definition in {scope_label}",
                subgrammar.name,
            )));
        }
    }
    let Some(start_kind) = symbols.get(&scope.start).copied() else {
        return Err(err(&format!(
            "start '{}' is not defined in {scope_label}; definitions are scope-local",
            scope.start,
        )));
    };
    if start_kind == ScopedSymbolKind::InternalTerminal {
        return Err(err(&format!(
            "start '{}' in {scope_label} is an internal terminal",
            scope.start,
        )));
    }
    Ok(symbols)
}

fn validate_ignore_terminal(grammar: &NamedGrammar, scope_label: &str) -> Result<(), GlrMaskError> {
    let Some(ignore_name) = grammar.ignore.as_deref() else {
        return Ok(());
    };
    let Some(rule) = grammar
        .rules
        .iter()
        .find(|rule| rule.name == ignore_name)
    else {
        return Err(err(&format!(
            "ignore terminal '{ignore_name}' is not defined in {scope_label}; definitions are scope-local",
        )));
    };
    if !rule.is_terminal || rule.is_internal {
        return Err(err(&format!(
            "ignore '{ignore_name}' in {scope_label} must name a local emitting terminal",
        )));
    }
    if matches!(rule.expr, GrammarExpr::SpecialToken(_)) {
        return Err(err(
            "a special LLM token terminal cannot be the ignore terminal",
        ));
    }
    let resolved = resolved_named_terminal_exprs(grammar)?;
    let expr = resolved.get(ignore_name).ok_or_else(|| {
        err(&format!(
            "ignore terminal '{ignore_name}' could not be resolved in {scope_label}",
        ))
    })?;
    if expr.is_nullable() {
        return Err(err(&format!(
            "ignore terminal '{ignore_name}' in {scope_label} must consume at least one byte",
        )));
    }
    for rule in grammar.rules.iter().filter(|rule| !rule.is_terminal) {
        if grammar_expr_contains_ref(&rule.expr, ignore_name) {
            return Err(err(&format!(
                "ignore terminal '{ignore_name}' is referenced explicitly by parser rule '{}' in {scope_label}; ignored terminals are implicit at grammar boundaries and between lexical atoms",
                rule.name,
            )));
        }
    }
    Ok(())
}

fn grammar_expr_contains_ref(expr: &GrammarExpr, name: &str) -> bool {
    match expr {
        GrammarExpr::Ref(reference) => reference == name,
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            grammar_expr_contains_ref(inner, name)
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            parts.iter().any(|part| grammar_expr_contains_ref(part, name))
        }
        GrammarExpr::Exclude { expr, exclude } => {
            grammar_expr_contains_ref(expr, name) || grammar_expr_contains_ref(exclude, name)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            grammar_expr_contains_ref(expr, name) || grammar_expr_contains_ref(intersect, name)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items
                .iter()
                .any(|(item, _)| grammar_expr_contains_ref(item, name))
                || grammar_expr_contains_ref(separator, name)
        }
        GrammarExpr::ExprNFA(expr_nfa) => expr_nfa
            .symbols
            .iter()
            .any(|symbol| grammar_expr_contains_ref(symbol, name)),
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::SpecialToken(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => false,
    }
}

struct FlattenContext {
    next_scope_id: usize,
    used_names: HashSet<String>,
    used_partition_names: HashSet<String>,
    rules: Vec<NamedRule>,
    lexer_partitions: BTreeMap<String, String>,
    lexer_literal_partition_constraints: Vec<(Vec<u8>, String)>,
}

impl FlattenContext {
    fn fresh_global_name(&mut self, base: String) -> String {
        if self.used_names.insert(base.clone()) {
            return base;
        }
        for suffix in 1usize.. {
            let candidate = format!("{base}_{suffix}");
            if self.used_names.insert(candidate.clone()) {
                return candidate;
            }
        }
        unreachable!()
    }

    fn reserve_exact_name(&mut self, name: &str) -> Result<String, GlrMaskError> {
        if !self.used_names.insert(name.to_string()) {
            return Err(err(&format!("duplicate top-level definition '{name}'")));
        }
        Ok(name.to_string())
    }

    fn fresh_partition_name(&mut self, base: String) -> String {
        if self.used_partition_names.insert(base.clone()) {
            return base;
        }
        for suffix in 1usize.. {
            let candidate = format!("{base}_{suffix}");
            if self.used_partition_names.insert(candidate.clone()) {
                return candidate;
            }
        }
        unreachable!()
    }
}

fn flatten_scoped_grammar(scope: ParsedGlrmScope) -> Result<NamedGrammar, GlrMaskError> {
    let mut context = FlattenContext {
        next_scope_id: 0,
        used_names: HashSet::new(),
        used_partition_names: HashSet::new(),
        rules: Vec::new(),
        lexer_partitions: BTreeMap::new(),
        lexer_literal_partition_constraints: Vec::new(),
    };
    let start = flatten_scope(&scope, true, "grammar", &mut context)?;
    let (lexer_partitions, lexer_literal_partitions) = canonicalize_flattened_lexer_partitions(
        &context.rules,
        &start,
        context.lexer_partitions,
        context.lexer_literal_partition_constraints,
    )?;
    Ok(NamedGrammar {
        rules: context.rules,
        start,
        ignore: None,
        lexer_partitions,
        lexer_literal_partitions,
        default_lexer_partition: None,
    })
}

fn canonicalize_flattened_lexer_partitions(
    rules: &[NamedRule],
    start: &str,
    lexer_partitions: BTreeMap<String, String>,
    literal_constraints: Vec<(Vec<u8>, String)>,
) -> Result<(BTreeMap<String, String>, BTreeMap<Vec<u8>, String>), GlrMaskError> {
    let grammar = NamedGrammar {
        rules: rules.to_vec(),
        start: start.to_string(),
        ignore: None,
        lexer_partitions: BTreeMap::new(),
        lexer_literal_partitions: BTreeMap::new(),
        default_lexer_partition: None,
    };
    let resolved_terminals = resolved_named_terminal_exprs(&grammar)?;
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    let mut connect = |left: &str, right: &str| {
        adjacency
            .entry(left.to_string())
            .or_default()
            .insert(right.to_string());
        adjacency
            .entry(right.to_string())
            .or_default()
            .insert(left.to_string());
    };

    let mut partition_by_expr = HashMap::new();
    for (terminal, partition) in &lexer_partitions {
        connect(partition, partition);
        let Some(expr) = resolved_terminals.get(terminal) else {
            continue;
        };
        if let Some(previous) = partition_by_expr.insert(expr.clone(), partition.clone()) {
            connect(&previous, partition);
        }
    }

    let mut partition_by_literal = BTreeMap::<Vec<u8>, String>::new();
    for (literal, partition) in &literal_constraints {
        connect(partition, partition);
        if let Some(previous) = partition_by_literal.insert(literal.clone(), partition.clone()) {
            connect(&previous, partition);
        }
    }

    let mut canonical = BTreeMap::<String, String>::new();
    let mut visited = BTreeSet::new();
    for partition in adjacency.keys() {
        if visited.contains(partition) {
            continue;
        }
        let mut stack = vec![partition.clone()];
        let mut component = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            component.push(current.clone());
            if let Some(neighbors) = adjacency.get(&current) {
                stack.extend(neighbors.iter().cloned());
            }
        }
        component.sort_unstable();
        let representative = component
            .first()
            .expect("partition component must be non-empty")
            .clone();
        for member in component {
            canonical.insert(member, representative.clone());
        }
    }

    let lexer_partitions = lexer_partitions
        .into_iter()
        .map(|(terminal, partition)| {
            let partition = canonical.get(&partition).cloned().unwrap_or(partition);
            (terminal, partition)
        })
        .collect();
    let mut lexer_literal_partitions = BTreeMap::new();
    for (literal, partition) in literal_constraints {
        let partition = canonical.get(&partition).cloned().unwrap_or(partition);
        if let Some(previous) = lexer_literal_partitions.insert(literal.clone(), partition.clone()) {
            debug_assert_eq!(previous, partition);
        }
    }
    Ok((lexer_partitions, lexer_literal_partitions))
}

fn flatten_scope(
    scope: &ParsedGlrmScope,
    top_level: bool,
    scope_label: &str,
    context: &mut FlattenContext,
) -> Result<String, GlrMaskError> {
    let symbol_kinds = scope_symbol_kinds(scope, scope_label)?;
    let local_grammar = named_grammar_for_scope(scope)?;
    validate_ignore_terminal(&local_grammar, scope_label)?;

    let original_emitting_terminals = symbol_kinds
        .iter()
        .filter_map(|(name, kind)| (*kind == ScopedSymbolKind::Terminal).then_some(name.clone()))
        .collect::<HashSet<_>>();
    let subgrammar_names = symbol_kinds
        .iter()
        .filter_map(|(name, kind)| (*kind == ScopedSymbolKind::Subgrammar).then_some(name.clone()))
        .collect::<HashSet<_>>();

    let scope_id = context.next_scope_id;
    context.next_scope_id += 1;

    let mut local_existing_names = symbol_kinds.keys().cloned().collect::<HashSet<_>>();
    let mut working_rules = scope.rules.clone();
    let entry_local_name = fresh_local_name(&mut local_existing_names, "__glrm_scope_entry");

    let mut promoted_default_partitions = BTreeMap::<String, String>::new();
    if let Some(default_partition) = scope.default_lexer_partition.as_deref() {
        let mut promoter = DefaultPartitionAtomPromoter {
            default_partition,
            explicit_literal_partitions: &local_grammar.lexer_literal_partitions,
            emitting_terminals: &original_emitting_terminals,
            existing_names: &mut local_existing_names,
            promoted: HashMap::new(),
            generated_rules: Vec::new(),
            generated_partitions: BTreeMap::new(),
        };
        for rule in working_rules.iter_mut().filter(|rule| !rule.is_terminal) {
            rule.expr = promoter.rewrite_expr(&rule.expr)?;
        }
        working_rules.extend(promoter.generated_rules);
        promoted_default_partitions = promoter.generated_partitions;
    }

    let emitting_terminals = working_rules
        .iter()
        .filter(|rule| rule.is_terminal && !rule.is_internal)
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();
    let byte_emitting_terminals = working_rules
        .iter()
        .filter(|rule| {
            rule.is_terminal
                && !rule.is_internal
                && !matches!(rule.expr, GrammarExpr::SpecialToken(_))
        })
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();

    if let Some(ignore_name) = scope.ignore.as_deref() {
        let skip_local_name = fresh_local_name(&mut local_existing_names, "__glrm_ignore_skip");
        working_rules.push(NamedRule {
            name: skip_local_name.clone(),
            expr: GrammarExpr::Choice(vec![
                GrammarExpr::Epsilon,
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(skip_local_name.clone()),
                    GrammarExpr::Ref(ignore_name.to_string()),
                ]),
            ]),
            is_terminal: false,
            is_internal: false,
        });

        let mut rewriter = IgnoreScopeRewriter {
            skip_name: &skip_local_name,
            emitting_terminals: &emitting_terminals,
            subgrammar_names: &subgrammar_names,
            existing_names: &mut local_existing_names,
            wrappers: HashMap::new(),
            generated_rules: Vec::new(),
        };
        for rule in working_rules.iter_mut().filter(|rule| !rule.is_terminal) {
            if rule.name == skip_local_name {
                continue;
            }
            rule.expr = rewriter.rewrite_expr(&rule.expr)?;
        }
        let entry_core = rewriter.rewrite_expr(&GrammarExpr::Ref(scope.start.clone()))?;
        let entry_expr = append_trailing_skip(entry_core, &skip_local_name);
        working_rules.extend(rewriter.generated_rules);
        working_rules.push(NamedRule {
            name: entry_local_name.clone(),
            expr: entry_expr,
            is_terminal: false,
            is_internal: false,
        });
    } else {
        working_rules.push(NamedRule {
            name: entry_local_name.clone(),
            expr: GrammarExpr::Ref(scope.start.clone()),
            is_terminal: false,
            is_internal: false,
        });
    }

    let original_names = symbol_kinds.keys().cloned().collect::<HashSet<_>>();
    let mut local_names = working_rules
        .iter()
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();
    local_names.extend(subgrammar_names.iter().cloned());
    let mut name_map = HashMap::<String, String>::new();
    let mut ordered_names = local_names.into_iter().collect::<Vec<_>>();
    ordered_names.sort_unstable();
    for local_name in ordered_names {
        let mapped = if top_level && original_names.contains(&local_name) {
            context.reserve_exact_name(&local_name)?
        } else {
            let base = if top_level {
                local_name.clone()
            } else {
                format!("__glrm_subgrammar_{scope_id}_{local_name}")
            };
            context.fresh_global_name(base)
        };
        name_map.insert(local_name, mapped);
    }

    merge_scope_lexer_config(
        scope,
        &local_grammar,
        &symbol_kinds,
        &byte_emitting_terminals,
        &promoted_default_partitions,
        &name_map,
        top_level,
        scope_id,
        scope_label,
        context,
    )?;

    for subgrammar in &scope.subgrammars {
        let child_label = format!("{scope_label}::{}", subgrammar.name);
        let child_entry = flatten_scope(&subgrammar.scope, false, &child_label, context)?;
        let alias_name = name_map
            .get(&subgrammar.name)
            .expect("subgrammar name must be allocated")
            .clone();
        context.rules.push(NamedRule {
            name: alias_name,
            expr: GrammarExpr::Ref(child_entry),
            is_terminal: false,
            is_internal: false,
        });
    }

    for rule in working_rules {
        let mapped_name = name_map
            .get(&rule.name)
            .expect("working rule name must be allocated")
            .clone();
        let mapped_expr = rewrite_scope_refs(&rule.expr, &name_map, scope_label)?;
        context.rules.push(NamedRule {
            name: mapped_name,
            expr: mapped_expr,
            is_terminal: rule.is_terminal,
            is_internal: rule.is_internal,
        });
    }

    Ok(name_map
        .get(&entry_local_name)
        .expect("scope entry must be allocated")
        .clone())
}

fn merge_scope_lexer_config(
    scope: &ParsedGlrmScope,
    local_grammar: &NamedGrammar,
    symbol_kinds: &BTreeMap<String, ScopedSymbolKind>,
    emitting_terminals: &HashSet<String>,
    promoted_default_partitions: &BTreeMap<String, String>,
    name_map: &HashMap<String, String>,
    top_level: bool,
    scope_id: usize,
    scope_label: &str,
    context: &mut FlattenContext,
) -> Result<(), GlrMaskError> {
    let mut local_partition_names = local_grammar
        .lexer_partitions
        .values()
        .chain(local_grammar.lexer_literal_partitions.values())
        .chain(scope.default_lexer_partition.iter())
        .chain(promoted_default_partitions.values())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut partition_name_map = BTreeMap::<String, String>::new();
    for partition in std::mem::take(&mut local_partition_names) {
        let base = if top_level {
            partition.clone()
        } else {
            format!("__glrm_subgrammar_{scope_id}_lexer_{partition}")
        };
        partition_name_map.insert(partition, context.fresh_partition_name(base));
    }
    let mapped_partition = |partition: &str| {
        partition_name_map
            .get(partition)
            .expect("scope-local partition must be allocated")
            .clone()
    };

    for terminal in emitting_terminals {
        let local_partition = local_grammar
            .lexer_partitions
            .get(terminal)
            .or_else(|| promoted_default_partitions.get(terminal))
            .or(scope.default_lexer_partition.as_ref());
        let Some(partition) = local_partition else {
            continue;
        };
        let mapped_terminal = name_map
            .get(terminal)
            .expect("local emitting terminal must be mapped")
            .clone();
        context
            .lexer_partitions
            .insert(mapped_terminal, mapped_partition(partition));
    }

    for (terminal, partition) in &local_grammar.lexer_partitions {
        if symbol_kinds.get(terminal) != Some(&ScopedSymbolKind::Terminal) {
            return Err(err(&format!(
                "lexer group in {scope_label} references non-local or non-emitting terminal '{terminal}'",
            )));
        }
        debug_assert_eq!(
            context
                .lexer_partitions
                .get(name_map.get(terminal).expect("local terminal must be mapped")),
            Some(&mapped_partition(partition)),
        );
    }

    if scope.default_lexer_partition.is_none() {
        for (literal, partition) in &local_grammar.lexer_literal_partitions {
            context
                .lexer_literal_partition_constraints
                .push((literal.clone(), mapped_partition(partition)));
        }
    }
    Ok(())
}

fn rewrite_scope_refs(
    expr: &GrammarExpr,
    name_map: &HashMap<String, String>,
    scope_label: &str,
) -> Result<GrammarExpr, GlrMaskError> {
    Ok(match expr {
        GrammarExpr::Ref(name) => GrammarExpr::Ref(
            name_map
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    err(&format!(
                        "definition '{name}' is not visible in {scope_label}; definitions are scope-local",
                    ))
                })?,
        ),
        GrammarExpr::Grouped(inner) => {
            GrammarExpr::Grouped(Box::new(rewrite_scope_refs(inner, name_map, scope_label)?))
        }
        GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
            parts
                .iter()
                .map(|part| rewrite_scope_refs(part, name_map, scope_label))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        GrammarExpr::Choice(options) => GrammarExpr::Choice(
            options
                .iter()
                .map(|option| rewrite_scope_refs(option, name_map, scope_label))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(rewrite_scope_refs(expr, name_map, scope_label)?),
            exclude: Box::new(rewrite_scope_refs(exclude, name_map, scope_label)?),
        },
        GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(rewrite_scope_refs(expr, name_map, scope_label)?),
            intersect: Box::new(rewrite_scope_refs(intersect, name_map, scope_label)?),
        },
        GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
            Box::new(rewrite_scope_refs(inner, name_map, scope_label)?),
            quantifier.clone(),
        ),
        GrammarExpr::SeparatedSequence {
            items,
            separator,
            allow_empty,
        } => GrammarExpr::SeparatedSequence {
            items: items
                .iter()
                .map(|(item, quantifier)| {
                    Ok((rewrite_scope_refs(item, name_map, scope_label)?, quantifier.clone()))
                })
                .collect::<Result<Vec<_>, GlrMaskError>>()?,
            separator: Box::new(rewrite_scope_refs(separator, name_map, scope_label)?),
            allow_empty: *allow_empty,
        },
        GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA::new(
            expr_nfa.nfa.clone(),
            expr_nfa
                .symbols
                .iter()
                .map(|symbol| rewrite_scope_refs(symbol, name_map, scope_label))
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::SpecialToken(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => expr.clone(),
    })
}

struct DefaultPartitionAtomPromoter<'a> {
    default_partition: &'a str,
    explicit_literal_partitions: &'a BTreeMap<Vec<u8>, String>,
    emitting_terminals: &'a HashSet<String>,
    existing_names: &'a mut HashSet<String>,
    promoted: HashMap<GrammarExpr, String>,
    generated_rules: Vec<NamedRule>,
    generated_partitions: BTreeMap<String, String>,
}

impl DefaultPartitionAtomPromoter<'_> {
    fn rewrite_expr(&mut self, expr: &GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Ref(_) | GrammarExpr::Epsilon => expr.clone(),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => self.promote_atom(expr.clone()),
            GrammarExpr::SpecialToken(_) => expr.clone(),
            GrammarExpr::Grouped(inner) => {
                GrammarExpr::Grouped(Box::new(self.rewrite_expr(inner)?))
            }
            GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
                parts
                    .iter()
                    .map(|part| self.rewrite_expr(part))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Choice(options) => GrammarExpr::Choice(
                options
                    .iter()
                    .map(|option| self.rewrite_expr(option))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. }
                if !self.contains_nonterminal_ref(expr) =>
            {
                self.promote_atom(expr.clone())
            }
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.rewrite_expr(expr)?),
                exclude: Box::new(self.rewrite_expr(exclude)?),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.rewrite_expr(expr)?),
                intersect: Box::new(self.rewrite_expr(intersect)?),
            },
            GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
                Box::new(self.rewrite_expr(inner)?),
                quantifier.clone(),
            ),
            GrammarExpr::SeparatedSequence {
                items,
                separator,
                allow_empty,
            } => GrammarExpr::SeparatedSequence {
                items: items
                    .iter()
                    .map(|(item, quantifier)| {
                        Ok((self.rewrite_expr(item)?, quantifier.clone()))
                    })
                    .collect::<Result<Vec<_>, GlrMaskError>>()?,
                separator: Box::new(self.rewrite_expr(separator)?),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA::new(
                expr_nfa.nfa.clone(),
                expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.rewrite_expr(symbol))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
        })
    }

    fn contains_nonterminal_ref(&self, expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(name) => !self.emitting_terminals.contains(name),
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                self.contains_nonterminal_ref(inner)
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                parts.iter().any(|part| self.contains_nonterminal_ref(part))
            }
            GrammarExpr::Exclude { expr, exclude } => {
                self.contains_nonterminal_ref(expr) || self.contains_nonterminal_ref(exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                self.contains_nonterminal_ref(expr) || self.contains_nonterminal_ref(intersect)
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items
                    .iter()
                    .any(|(item, _)| self.contains_nonterminal_ref(item))
                    || self.contains_nonterminal_ref(separator)
            }
            GrammarExpr::ExprNFA(expr_nfa) => expr_nfa
                .symbols
                .iter()
                .any(|symbol| self.contains_nonterminal_ref(symbol)),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    fn promote_atom(&mut self, atom: GrammarExpr) -> GrammarExpr {
        if let Some(name) = self.promoted.get(&atom) {
            return GrammarExpr::Ref(name.clone());
        }
        let partition = match &atom {
            GrammarExpr::Literal(bytes) => self
                .explicit_literal_partitions
                .get(bytes)
                .map(String::as_str)
                .unwrap_or(self.default_partition),
            _ => self.default_partition,
        };
        let name = fresh_local_name(self.existing_names, "__glrm_lexer_atom");
        self.generated_rules.push(NamedRule {
            name: name.clone(),
            expr: atom.clone(),
            is_terminal: true,
            is_internal: false,
        });
        self.generated_partitions
            .insert(name.clone(), partition.to_string());
        self.promoted.insert(atom, name.clone());
        GrammarExpr::Ref(name)
    }
}

struct IgnoreScopeRewriter<'a> {
    skip_name: &'a str,
    emitting_terminals: &'a HashSet<String>,
    subgrammar_names: &'a HashSet<String>,
    existing_names: &'a mut HashSet<String>,
    wrappers: HashMap<GrammarExpr, String>,
    generated_rules: Vec<NamedRule>,
}

impl IgnoreScopeRewriter<'_> {
    fn rewrite_expr(&mut self, expr: &GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Ref(name)
                if self.emitting_terminals.contains(name) || self.subgrammar_names.contains(name) =>
            {
                self.wrap_atom(expr.clone())
            }
            GrammarExpr::Ref(_) | GrammarExpr::Epsilon => expr.clone(),
            GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => self.wrap_atom(expr.clone()),
            GrammarExpr::Grouped(inner) => {
                GrammarExpr::Grouped(Box::new(self.rewrite_expr(inner)?))
            }
            GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
                parts
                    .iter()
                    .map(|part| self.rewrite_expr(part))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Choice(options) => GrammarExpr::Choice(
                options
                    .iter()
                    .map(|option| self.rewrite_expr(option))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. }
                if !self.contains_nonterminalish_ref(expr) =>
            {
                self.wrap_atom(expr.clone())
            }
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.rewrite_expr(expr)?),
                exclude: Box::new(self.rewrite_expr(exclude)?),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.rewrite_expr(expr)?),
                intersect: Box::new(self.rewrite_expr(intersect)?),
            },
            GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
                Box::new(self.rewrite_expr(inner)?),
                quantifier.clone(),
            ),
            GrammarExpr::SeparatedSequence {
                items,
                separator,
                allow_empty,
            } => GrammarExpr::SeparatedSequence {
                items: items
                    .iter()
                    .map(|(item, quantifier)| {
                        Ok((self.rewrite_expr(item)?, quantifier.clone()))
                    })
                    .collect::<Result<Vec<_>, GlrMaskError>>()?,
                separator: Box::new(self.rewrite_expr(separator)?),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA::new(
                expr_nfa.nfa.clone(),
                expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.rewrite_expr(symbol))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
        })
    }

    fn contains_nonterminalish_ref(&self, expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(name) => !self.emitting_terminals.contains(name),
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                self.contains_nonterminalish_ref(inner)
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                parts.iter().any(|part| self.contains_nonterminalish_ref(part))
            }
            GrammarExpr::Exclude { expr, exclude } => {
                self.contains_nonterminalish_ref(expr)
                    || self.contains_nonterminalish_ref(exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                self.contains_nonterminalish_ref(expr)
                    || self.contains_nonterminalish_ref(intersect)
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items
                    .iter()
                    .any(|(item, _)| self.contains_nonterminalish_ref(item))
                    || self.contains_nonterminalish_ref(separator)
            }
            GrammarExpr::ExprNFA(expr_nfa) => expr_nfa
                .symbols
                .iter()
                .any(|symbol| self.contains_nonterminalish_ref(symbol)),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    fn wrap_atom(&mut self, atom: GrammarExpr) -> GrammarExpr {
        if let Some(name) = self.wrappers.get(&atom) {
            return GrammarExpr::Ref(name.clone());
        }
        let name = fresh_local_name(self.existing_names, "__glrm_ignored_atom");
        self.generated_rules.push(NamedRule {
            name: name.clone(),
            expr: GrammarExpr::Sequence(vec![
                GrammarExpr::Ref(self.skip_name.to_string()),
                atom.clone(),
            ]),
            is_terminal: false,
            is_internal: false,
        });
        self.wrappers.insert(atom, name.clone());
        GrammarExpr::Ref(name)
    }
}

fn append_trailing_skip(expr: GrammarExpr, skip_name: &str) -> GrammarExpr {
    let skip = GrammarExpr::Ref(skip_name.to_string());
    match expr {
        GrammarExpr::Epsilon => skip,
        GrammarExpr::Sequence(mut parts) => {
            parts.push(skip);
            GrammarExpr::Sequence(parts)
        }
        other => GrammarExpr::Sequence(vec![other, skip]),
    }
}

fn fresh_local_name(existing_names: &mut HashSet<String>, base: &str) -> String {
    if existing_names.insert(base.to_string()) {
        return base.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if existing_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}

// ---- Parse a byte character class string ----------------------------------

// ---- Small helpers ---------------------------------------------------------

fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(err(&format!("invalid hex digit '{}'", b as char))),
    }
}

fn ensure_nfa_state(nfa: &mut NFA, state: u32) {
    while nfa.states.len() <= state as usize {
        nfa.add_state();
    }
}

fn intern_expr_nfa_symbol(
    symbols: &mut Vec<GrammarExpr>,
    symbol_labels: &mut HashMap<GrammarExpr, Label>,
    symbol: GrammarExpr,
) -> Label {
    if let Some(&label) = symbol_labels.get(&symbol) {
        return label;
    }
    let label = i32::try_from(symbols.len()).expect("ExprNFA symbol table exceeded i32 labels");
    symbols.push(symbol.clone());
    symbol_labels.insert(symbol, label);
    label
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
    use crate::grammar::ast::lower;
    use crate::grammar::flat::Symbol;

    fn single_path_terminal_names(
        lowered: &crate::grammar::flat::GrammarDef,
        symbol: &Symbol,
    ) -> Vec<String> {
        match symbol {
            Symbol::Terminal(id) => vec![lowered.terminal_display_name(*id)],
            Symbol::Nonterminal(id) => {
                let rules = lowered
                    .rules
                    .iter()
                    .filter(|rule| rule.lhs == *id)
                    .collect::<Vec<_>>();
                assert_eq!(rules.len(), 1, "expected a single-path helper nonterminal");
                rules[0]
                    .rhs
                    .iter()
                    .flat_map(|child| single_path_terminal_names(lowered, child))
                    .collect()
            }
        }
    }

    #[test]
    fn parses_named_expr_nfa_definition() {
        let grammar = from_glrm(
            r#"
start obj;

fa obj ::= {
start 0;
accept 4;

0 -- "\"name\": " --> 1;
1 -- "," "\"email\": " --> 2;
1 -- "," "\"description\": " --> 3;
2 -- "," "\"thumbnail\": " --> 3;
2 --> 4;
3 --> 4;
};
"#,
        )
        .unwrap();

        assert_eq!(grammar.rules.len(), 1);
        assert!(matches!(grammar.rules[0].expr, GrammarExpr::ExprNFA(_)));
        lower(&grammar).unwrap();
    }

    #[test]
    fn dumps_expr_nfa_as_own_definition() {
        let grammar = from_glrm(
            r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- "a" --> 1;
};
"#,
        )
        .unwrap();
        let dumped = to_glrm(&grammar);
        assert!(dumped.contains("fa obj ::= {"), "{dumped}");
        assert!(dumped.contains("  start 0;"), "{dumped}");
        assert!(dumped.contains("  accept 1;"), "{dumped}");
        assert!(dumped.contains("  0 -- \"a\" --> 1;"), "{dumped}");
        assert!(!dumped.contains("ExprNFA("), "{dumped}");
    }

    #[test]
    fn special_llm_token_atom_roundtrips() {
        let grammar = from_glrm(
            r#"
                start start;
                t END ::= @token(128009);
                nt start ::= "a" END @token(42);
            "#,
        )
        .unwrap();
        let dumped = to_glrm(&grammar);
        assert!(dumped.contains("@token(128009)"), "{dumped}");
        assert!(dumped.contains("@token(42)"), "{dumped}");
        assert_eq!(from_glrm(&dumped).unwrap().rules, grammar.rules);
    }

    #[test]
    fn expr_nfa_transition_symbols_accept_full_expressions() {
        let grammar = from_glrm(
            r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- [a-z] - "x" --> 1;
};
"#,
        )
        .unwrap();
        let GrammarExpr::ExprNFA(expr_nfa) = &grammar.rules[0].expr else {
            panic!("expected ExprNFA rule");
        };
        assert!(matches!(
            expr_nfa.symbols.first(),
            Some(GrammarExpr::Exclude { .. })
        ));
    }

    #[test]
    fn expr_nfa_transition_symbols_reject_raw_regex_literals() {
        let err = from_glrm(
            r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- /[a-z]+/ --> 1;
};
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("raw regex literals are only allowed in terminal (`t`) rules"),
            "{err}"
        );
    }

    #[test]
    fn exclude_rhs_sequence_requires_parentheses() {
        let err = from_glrm(
            r#"
start z;
nt A ::= a b | c d | e f;
nt z ::= x (A - c d);
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("RHS sequence subtraction must be parenthesized"), "{err}");
    }

    #[test]
    fn grouped_exclude_rhs_preserves_parenthesized_ref() {
        let grammar = from_glrm(
            r#"
start z;
nt A ::= a b | c d | e f;
nt B ::= c d | e f;
nt z ::= x (A - B) | x (A - (B));
"#,
        )
        .unwrap();
        let GrammarExpr::Choice(options) = &grammar.rules[2].expr else {
            panic!("expected choice");
        };
        assert!(matches!(
            options[0],
            GrammarExpr::Sequence(_)
        ));
        let GrammarExpr::Sequence(second_parts) = &options[1] else {
            panic!("expected sequence");
        };
        let GrammarExpr::Exclude { exclude, .. } = &second_parts[1] else {
            panic!("expected exclude expr");
        };
        assert!(matches!(exclude.as_ref(), GrammarExpr::Grouped(_)));
    }

    #[test]
    fn lowering_subtracts_exact_nonterminal_alternatives() {
        let grammar = from_glrm(
            r#"
start z;
nt A ::= "a" "b" | "c" "d" | "e" "f";
nt B ::= "c" "d" | "e" "f";
nt z ::= "x" (A - B);
"#,
        )
        .unwrap();

        let lowered = lower(&grammar).unwrap();
        let z_rule = lowered
            .rules
            .iter()
            .find(|rule| rule.lhs == lowered.start)
            .expect("start rule should exist");
        assert_eq!(z_rule.rhs.len(), 2);

        let Symbol::Nonterminal(filtered_nt) = z_rule.rhs[1] else {
            panic!("expected filtered nonterminal");
        };
        let filtered_rules = lowered
            .rules
            .iter()
            .filter(|rule| rule.lhs == filtered_nt)
            .collect::<Vec<_>>();
        assert_eq!(filtered_rules.len(), 1);
        assert_eq!(filtered_rules[0].rhs.len(), 2);

        let filtered_terminals = filtered_rules[0]
            .rhs
            .iter()
            .flat_map(|symbol| single_path_terminal_names(&lowered, symbol))
            .collect::<Vec<_>>();
        assert_eq!(filtered_terminals, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn lowering_accepts_parenthesized_ref_exact_subtraction() {
        let grammar = from_glrm(
            r#"
start z;
nt A ::= "a" "b" | "c" "d" | "e" "f";
nt B ::= "c" "d" | "e" "f";
nt z ::= "x" (A - (B));
"#,
        )
        .unwrap();

        lower(&grammar).unwrap();
    }


    #[test]
    fn sepseq_rejects_stacked_item_quantifiers() {
        let err = from_glrm(
            r#"
start s;
nt s ::= "," ~+ ( item+? );
nt item ::= "a";
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("only one postfix quantifier may bind"), "{err}");
    }

    #[test]
    fn sepseq_allows_grouped_inner_quantifier_then_outer_item_quantifier() {
        let grammar = from_glrm(
            r#"
start s;
nt s ::= "," ~+ ( (item+)? );
nt item ::= "a";
"#,
        )
        .unwrap();

        let GrammarExpr::SeparatedSequence { items, .. } = &grammar.rules[0].expr else {
            panic!("expected separated sequence");
        };
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0].1, Some(Quantifier::Optional)));
        assert!(matches!(items[0].0, GrammarExpr::Grouped(_)));
    }

    #[test]
    fn rejects_nested_expr_nfa_at_lowering() {
        let nfa_rule = from_glrm(
            r#"
start inner;
fa inner ::= {
start 0;
accept 1;
0 -- "a" --> 1;
};
"#,
        )
        .unwrap()
        .rules
        .into_iter()
        .next()
        .unwrap();

        let grammar = NamedGrammar {
            rules: vec![NamedRule {
                name: "start".to_string(),
                expr: GrammarExpr::Sequence(vec![nfa_rule.expr, GrammarExpr::Literal(b"b".to_vec())]),
                is_terminal: false,
                is_internal: false,
            }],
            start: "start".to_string(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };

        let err = lower(&grammar).unwrap_err().to_string();
        assert!(err.contains("complete expression of a nonterminal rule"), "{err}");
    }

    #[test]
    fn lexer_groups_round_trip_and_control_tokenizer_partitions() {
        let grammar = from_glrm(
            r#"
start s;
lexer group words ::= A, B;
t A ::= "a";
t B ::= "ab";
t C ::= "z";
nt s ::= A | B | C;
"#,
        )
        .unwrap();
        assert_eq!(grammar.lexer_partitions.get("A").map(String::as_str), Some("words"));
        assert_eq!(grammar.lexer_partitions.get("B").map(String::as_str), Some("words"));
        assert!(!grammar.lexer_partitions.contains_key("C"));

        let dumped = to_glrm(&grammar);
        assert!(dumped.contains("lexer group words ::= A, B;"), "{dumped}");
        let reparsed = from_glrm(&dumped).unwrap();
        assert_eq!(reparsed.lexer_partitions, grammar.lexer_partitions);

        let lowered = lower(&grammar).unwrap();
        assert_eq!(lowered.lexer_partitions.len(), 2);
        let tokenizer = crate::compiler::pipeline::build_tokenizer_with_partition_options(
            &lowered,
            false,
            false,
        );
        assert!(tokenizer.has_epsilon_transitions());
        assert_eq!(
            tokenizer.initial_epsilon_branch_count(),
            2,
            "A/B should share one component while unspecified C is isolated in stress mode",
        );
    }

    #[test]
    fn lexer_groups_accept_anonymous_literals_and_a_catch_all() {
        let grammar = from_glrm(
            r#"
start s;
lexer group punctuation ::= "{";
lexer group literals ::= @literals;
lexer group patterns ::= *;
t WORD ::= /[a-z]+/;
nt s ::= "{" WORD "}";
"#,
        )
        .unwrap();
        assert_eq!(
            grammar
                .lexer_literal_partitions
                .get(b"{".as_slice())
                .map(String::as_str),
            Some("punctuation"),
        );
        assert_eq!(
            grammar
                .lexer_literal_partitions
                .get(b"}".as_slice())
                .map(String::as_str),
            Some("literals"),
        );
        assert_eq!(grammar.default_lexer_partition.as_deref(), Some("patterns"));

        let dumped = to_glrm(&grammar);
        assert!(
            dumped.contains("lexer group punctuation ::= \"{\";"),
            "{dumped}",
        );
        assert!(
            dumped.contains("lexer group literals ::= @literals;"),
            "{dumped}",
        );
        assert!(dumped.contains("lexer group patterns ::= *;"), "{dumped}");
        let reparsed = from_glrm(&dumped).unwrap();
        assert_eq!(
            reparsed.lexer_literal_partitions,
            grammar.lexer_literal_partitions,
        );
        assert_eq!(
            reparsed.default_lexer_partition,
            grammar.default_lexer_partition,
        );

        let lowered = lower(&grammar).unwrap();
        assert_eq!(lowered.lexer_partitions.len(), lowered.terminals.len());
        let punctuation = lowered
            .lexer_partitions
            .values()
            .filter(|partition| partition.as_str() == "punctuation")
            .count();
        let patterns = lowered
            .lexer_partitions
            .values()
            .filter(|partition| partition.as_str() == "patterns")
            .count();
        assert_eq!(punctuation, 1);
        assert_eq!(patterns, 1);
        assert_eq!(
            lowered
                .lexer_partitions
                .values()
                .filter(|partition| partition.as_str() == "literals")
                .count(),
            1,
        );
    }

    #[test]
    fn lexer_group_assignments_follow_deduplicated_terminal_aliases() {
        let grammar = from_glrm(
            r#"
start s;
lexer group same ::= A, B;
t A ::= "a";
t B ::= "a";
nt s ::= A | B;
"#,
        )
        .unwrap();
        let lowered = lower(&grammar).unwrap();
        assert_eq!(lowered.terminals.len(), 1);
        assert_eq!(lowered.lexer_partitions.len(), 1);
        assert_eq!(
            lowered.lexer_partitions.values().next().map(String::as_str),
            Some("same"),
        );
    }

    #[test]
    fn conflicting_groups_for_deduplicated_terminal_aliases_are_rejected() {
        let grammar = from_glrm(
            r#"
start s;
lexer group left ::= A;
lexer group right ::= B;
t A ::= "a";
t B ::= "a";
nt s ::= A | B;
"#,
        )
        .unwrap();
        let error = lower(&grammar).unwrap_err().to_string();
        assert!(error.contains("both lexer groups"), "{error}");
    }

    #[test]
    fn named_grammar_isolate_terminal_is_lowered_to_a_partition() {
        let mut grammar = from_glrm(
            r#"
start s;
t A ::= "a";
t B ::= "b";
nt s ::= A B;
"#,
        )
        .unwrap();
        grammar.isolate_terminal("B");
        let lowered = lower(&grammar).unwrap();
        assert_eq!(lowered.lexer_partitions.len(), 1);
        let b_id = lowered
            .terminal_names
            .iter()
            .find_map(|(&id, name)| (name == "B").then_some(id))
            .unwrap();
        assert_eq!(
            lowered.lexer_partitions.get(&b_id).map(String::as_str),
            Some("__isolated_B"),
        );
    }

    #[test]
    fn named_grammar_isolate_terminal_avoids_existing_partition_name() {
        let mut grammar = from_glrm(
            r#"
start s;
t A ::= "a";
t B ::= "b";
nt s ::= A B;
"#,
        )
        .unwrap();
        grammar.set_lexer_partition("__isolated_B", ["A"]);
        grammar.isolate_terminal("B");
        assert_eq!(
            grammar.lexer_partitions.get("B").map(String::as_str),
            Some("__isolated_B_2"),
        );
    }

    #[test]
    fn lexer_group_rejects_unknown_terminal_at_lowering() {
        let grammar = from_glrm(
            r#"
start s;
lexer group bad ::= MISSING;
t A ::= "a";
nt s ::= A;
"#,
        )
        .unwrap();
        let error = lower(&grammar).unwrap_err().to_string();
        assert!(error.contains("unknown or non-emitting terminal 'MISSING'"), "{error}");
    }

    #[test]
    fn flattened_subgrammar_dump_reparses_to_the_same_flat_grammar() {
        let grammar = from_glrm(
            r#"
start document;
ignore WS;
t WS ::= " "+;

g inner ::= {
    start value;
    ignore NL;
    t NL ::= "\n"+;
    nt value ::= "a" "b";
};

nt document ::= "<" inner ">";
"#,
        )
        .unwrap();
        let dumped = to_glrm(&grammar);
        let reparsed = from_glrm(&dumped).unwrap();
        assert_eq!(
            serde_json::to_value(lower(&grammar).unwrap()).unwrap(),
            serde_json::to_value(lower(&reparsed).unwrap()).unwrap(),
            "dumped flattened grammar:\n{dumped}",
        );
    }

    #[test]
    fn subgrammar_lexer_catch_all_is_scope_local_after_flattening() {
        let grammar = from_glrm(
            r#"
start document;
lexer group outer ::= *;
t OUTER ::= "x";

g inner ::= {
    start value;
    lexer group inner ::= *;
    t INNER ::= "a";
    nt value ::= INNER [bc];
};

nt document ::= OUTER inner;
"#,
        )
        .unwrap();

        assert!(grammar.default_lexer_partition.is_none());
        assert_eq!(grammar.lexer_partitions.get("OUTER").map(String::as_str), Some("outer"));
        for (terminal, partition) in &grammar.lexer_partitions {
            if terminal.starts_with("__glrm_subgrammar_") {
                assert_ne!(partition, "outer", "{terminal} inherited the outer catch-all");
            }
        }
        assert!(
            grammar
                .lexer_partitions
                .iter()
                .any(|(terminal, partition)| terminal.starts_with("__glrm_subgrammar_")
                    && partition.contains("_lexer_inner")),
            "expected a private child lexer partition: {:?}",
            grammar.lexer_partitions,
        );
    }

    #[test]
    fn identical_terminals_in_independent_catch_all_scopes_coalesce_partition_groups() {
        let grammar = from_glrm(
            r#"
start document;
lexer group outer ::= *;
t A ::= "a";

g inner ::= {
    start value;
    lexer group inner ::= *;
    t A ::= "a";
    nt value ::= A;
};

nt document ::= A inner;
"#,
        )
        .unwrap();

        let lowered = lower(&grammar).unwrap();
        assert_eq!(lowered.terminals.len(), 1, "identical terminal languages should still deduplicate");
        assert_eq!(lowered.lexer_partitions.len(), 1);
    }
}
