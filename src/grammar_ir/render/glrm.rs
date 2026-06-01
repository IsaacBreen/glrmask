//! GLRM rendering for named grammar IR.
//!
//! This is serialization/diagnostic output. Parsing lives in
//! `grammar_ir::glrm`; rendering must not lower to `GrammarDef`.

use crate::grammar_ir::ast::{GrammarExpr, NamedGrammar};
use crate::grammar_ir::expr_nfa::ExprNFA;
use crate::automata::unweighted::dfa::Label;

pub fn to_glrm(grammar: &NamedGrammar) -> String {
    let mut out = String::new();
    out.push_str(&format!("start {};\n", grammar.start));
    if let Some(ref ign) = grammar.ignore {
        out.push_str(&format!("ignore {};\n", ign));
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
        GrammarExpr::Grouped(inner) => format!("({})", dump_nt_expr(inner, false)),
        GrammarExpr::Literal(bytes) => format!("\"{}\"", escape_bytes_for_string(bytes)),
        GrammarExpr::RawRegex(pat) => format!("/{}/", escape_regex_for_slash(pat)),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let inner = if *negate { format!("^{}", def) } else { def.clone() };
            let suffix = if *utf8 { "/utf8" } else { "" };
            format!("[{}]{}", inner, suffix)
        }
        GrammarExpr::LexerDfa(dfa) => format!("LexerDfa(states={})", dfa.num_states()),
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
                .map(|(e, req)| {
                    let s = dump_nt_postfix(e);
                    if *req { s } else { format!("{}?", s) }
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
