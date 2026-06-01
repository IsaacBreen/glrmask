//! Lark-like rendering for named grammar expressions.
//!
//! This file is purely observational: it formats grammar IR for humans and for
//! diagnostics. It must not allocate terminal ids, create productions, or perform
//! lowering.

use crate::sets::byte_set::U8Set;
use crate::grammar_ir::ast::{GrammarExpr, NamedGrammar};

/// Render a named grammar in a Lark-like human-readable format.
pub fn to_lark(grammar: &NamedGrammar) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    writeln!(out, "// start: {}", grammar.start).unwrap();
    if let Some(ref ign) = grammar.ignore {
        writeln!(out, "%ignore {}", ign).unwrap();
    }
    writeln!(out).unwrap();

    let terminals: Vec<_> = grammar.rules.iter().filter(|r| r.is_terminal).collect();
    let nonterminals: Vec<_> = grammar.rules.iter().filter(|r| !r.is_terminal).collect();

    if !nonterminals.is_empty() {
        writeln!(out, "// === Nonterminal rules ===").unwrap();
        for rule in &nonterminals {
            let prefix = if rule.is_internal { "// [internal] " } else { "" };
            write!(out, "{}{}: ", prefix, rule.name).unwrap();
            grammar_expr_to_lark(&rule.expr, &mut out, false);
            writeln!(out).unwrap();
        }
        writeln!(out).unwrap();
    }

    if !terminals.is_empty() {
        writeln!(out, "// === Terminal rules ===").unwrap();
        for rule in &terminals {
            let prefix = if rule.is_internal { "// [internal] " } else { "" };
            write!(out, "{}{}: ", prefix, rule.name).unwrap();
            grammar_expr_to_lark(&rule.expr, &mut out, false);
            writeln!(out).unwrap();
        }
    }

    out
}

/// Format a `GrammarExpr` in Lark-like syntax. `parens` controls whether
/// compound expressions get wrapped in parentheses for disambiguation.
pub(crate) fn grammar_expr_to_lark(expr: &GrammarExpr, out: &mut String, parens: bool) {
    grammar_expr_to_lark_with_indent(expr, out, parens, 0);
}

fn grammar_expr_to_lark_with_indent(
    expr: &GrammarExpr,
    out: &mut String,
    parens: bool,
    indent: usize,
) {
    use std::fmt::Write;
    match expr {
        GrammarExpr::Ref(name) => {
            out.push_str(name);
        }
        GrammarExpr::Grouped(inner) => {
            out.push('(');
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            out.push(')');
        }
        GrammarExpr::Sequence(items) => {
            if parens && items.len() > 1 {
                out.push('(');
            }
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
            }
            if parens && items.len() > 1 {
                out.push(')');
            }
        }
        GrammarExpr::Choice(alts) => {
            let multiline = alts.len() > 6;
            if parens && alts.len() > 1 {
                out.push('(');
            }
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    if multiline {
                        out.push('\n');
                        for _ in 0..(indent + 4) {
                            out.push(' ');
                        }
                        out.push_str("| ");
                    } else {
                        out.push_str(" | ");
                    }
                }
                let child_indent = if multiline { indent + 6 } else { indent };
                grammar_expr_to_lark_with_indent(alt, out, true, child_indent);
            }
            if parens && alts.len() > 1 {
                if multiline {
                    out.push('\n');
                    for _ in 0..indent {
                        out.push(' ');
                    }
                }
                out.push(')');
            }
        }
        GrammarExpr::Literal(bytes) => {
            // Try UTF-8 first; fall back to hex
            if let Ok(s) = std::str::from_utf8(bytes) {
                write!(out, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")).unwrap();
            } else {
                let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                write!(out, "/*hex:{}*/", hex_str).unwrap();
            }
        }
        GrammarExpr::Optional(inner) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('?');
        }
        GrammarExpr::Repeat(inner) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('*');
        }
        GrammarExpr::RepeatOne(inner) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('+');
        }
        GrammarExpr::RepeatRange { expr: inner, min, max } => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            write!(out, "~{}..{}", min, max).unwrap();
        }
        GrammarExpr::Epsilon => {
            out.push_str("/*eps*/");
        }
        GrammarExpr::Exclude { expr: inner, exclude } => {
            write!(out, "/*Exclude(").unwrap();
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            write!(out, " \\ ").unwrap();
            grammar_expr_to_lark_with_indent(exclude, out, false, indent);
            write!(out, ")*/").unwrap();
        }
        GrammarExpr::Intersect { expr: inner, intersect } => {
            write!(out, "/*Intersect(").unwrap();
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            write!(out, " & ").unwrap();
            grammar_expr_to_lark_with_indent(intersect, out, false, indent);
            write!(out, ")*/").unwrap();
        }
        GrammarExpr::CharClass { def, negate, utf8 } => {
            if *negate {
                write!(out, "[^{}]", def).unwrap();
            } else {
                write!(out, "[{}]", def).unwrap();
            }
            if *utf8 {
                write!(out, "/*utf8*/").unwrap();
            }
        }
        GrammarExpr::RawRegex(pattern) => {
            write!(out, "/{}/", pattern).unwrap();
        }
        GrammarExpr::LexerDfa(dfa) => {
            write!(out, "/*LexerDfa(states={})*/", dfa.num_states()).unwrap();
        }
        GrammarExpr::AnyByte => {
            out.push_str("/./ /*AnyByte*/");
        }
        GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
            write!(out, "/*SeparatedSequence(sep=").unwrap();
            grammar_expr_to_lark_with_indent(separator, out, false, indent);
            write!(out, ", allow_empty={}, items=[", allow_empty).unwrap();
            for (i, (item, required)) in items.iter().enumerate() {
                if i > 0 { write!(out, ", ").unwrap(); }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
                if !required { write!(out, "?").unwrap(); }
            }
            write!(out, "])*/").unwrap();
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            write!(
                out,
                "/*ExprNFA(states={}, symbols={})*/",
                expr_nfa.nfa.states.len(),
                expr_nfa.symbols.len()
            )
            .unwrap();
        }
    }
}

/// Encode a [`U8Set`] as a character-class definition string (without the surrounding `[...]`).
///
/// Uses range notation where possible. Always produces a non-negated form.
pub(crate) fn u8set_to_class_def(set: &U8Set) -> String {
    let mut out = String::new();
    let bytes: Vec<u8> = set.iter().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        let start = bytes[i];
        let mut end = start;
        i += 1;
        while i < bytes.len() && bytes[i] == end.wrapping_add(1) && end < 255 {
            end = bytes[i];
            i += 1;
        }
        push_class_char(&mut out, start);
        if end != start {
            if end == start + 1 {
                push_class_char(&mut out, end);
            } else {
                out.push('-');
                push_class_char(&mut out, end);
            }
        }
    }
    out
}

fn push_class_char(out: &mut String, b: u8) {
    use std::fmt::Write;
    match b {
        b'\\' => out.push_str("\\\\"),
        b']' => out.push_str("\\]"),
        b'-' => out.push_str("\\-"),
        b'^' => out.push_str("\\^"),
        0x20..=0x7E => out.push(b as char),
        _ => write!(out, "\\x{:02X}", b).unwrap(),
    }
}

pub(crate) fn escape_byte(b: u8) -> String {
    match b {
        b'\n' => "\\n".into(),
        b'\r' => "\\r".into(),
        b'\t' => "\\t".into(),
        b'\\' => "\\\\".into(),
        b'"' => "\\\"".into(),
        byte if byte.is_ascii_graphic() || byte == b' ' => (byte as char).to_string(),
        byte => format!("\\x{byte:02x}"),
    }
}

pub(crate) fn regex_escape_byte(b: u8) -> String {
    match b {
        b'.' | b'+' | b'*' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'|' | b'^' | b'$' | b'\\' => {
            format!("\\{}", b as char)
        }
        _ => escape_byte(b),
    }
}
