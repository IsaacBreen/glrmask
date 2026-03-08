//! NOTE: tokenizer regex parsing and construction stay in this split-out file
//! so `Tokenizer` remains focused on runtime stepping.
// SEP1_MAP: sep1 does not keep this as one standalone file; the nearest pieces are spread across `interface/tokenizer_combinators.rs`, `interface/interface.rs`, and regex builders in `finite_automata.rs`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::regex::{Expr, ExprGroup};
use crate::compiler::grammar_def::{GrammarDef, TerminalID};
use crate::ds::u8set::U8Set;

use super::tokenizer::Tokenizer;

fn literal_from_expr(expr: &Expr) -> Option<Vec<u8>> {
    match expr {
        Expr::U8Seq(bytes) => Some(bytes.clone()),
        Expr::Shared(inner) => literal_from_expr(inner),
        Expr::Epsilon => Some(Vec::new()),
        _ => None,
    }
}

fn finite_literals_from_expr(expr: &Expr, cap: usize) -> Option<Vec<Vec<u8>>> {
    fn product(left: Vec<Vec<u8>>, right: Vec<Vec<u8>>, cap: usize) -> Option<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        for lhs in &left {
            for rhs in &right {
                let mut bytes = lhs.clone();
                bytes.extend(rhs);
                out.push(bytes);
                if out.len() > cap {
                    return None;
                }
            }
        }
        Some(out)
    }

    match expr {
        Expr::U8Seq(bytes) => Some(vec![bytes.clone()]),
        Expr::U8Class(set) => {
            let out: Vec<Vec<u8>> = set.iter().map(|byte| vec![byte]).collect();
            (out.len() <= cap).then_some(out)
        }
        Expr::Seq(parts) => {
            let mut current = vec![Vec::new()];
            for part in parts {
                current = product(current, finite_literals_from_expr(part, cap)?, cap)?;
            }
            Some(current)
        }
        Expr::Choice(options) => {
            let mut out = Vec::new();
            for option in options {
                out.extend(finite_literals_from_expr(option, cap)?);
                if out.len() > cap {
                    return None;
                }
            }
            Some(out)
        }
        Expr::Repeat { expr, min, max } => {
            let Some(max) = max else {
                return None;
            };
            let inner = finite_literals_from_expr(expr, cap)?;
            let mut out = Vec::new();
            for count in *min..=*max {
                let mut variants = vec![Vec::new()];
                for _ in 0..count {
                    variants = product(variants, inner.clone(), cap)?;
                }
                out.extend(variants);
                if out.len() > cap {
                    return None;
                }
            }
            Some(out)
        }
        Expr::Shared(inner) => finite_literals_from_expr(inner, cap),
        Expr::Epsilon => Some(vec![Vec::new()]),
    }
}

fn build_literal_tokenizer(terminals: &[(TerminalID, Vec<u8>, bool)]) -> Tokenizer {
    let num_terminals = terminals
        .iter()
        .map(|(terminal, _, _)| *terminal)
        .max()
        .map(|terminal| terminal as usize + 1)
        .unwrap_or(0);

    let mut dfa = crate::automata::dfa::DFA::new(1);
    dfa.ensure_group_capacity(num_terminals);

    let mut prefixes = std::collections::BTreeMap::<Vec<u8>, u32>::new();
    prefixes.insert(Vec::new(), 0);

    for (terminal, bytes, is_non_greedy) in terminals {
        let mut state = 0u32;
        let mut byte_set = U8Set::empty();

        // Mark the start state (and all intermediate prefix states) as having
        // this terminal reachable in the future.  The previous code only marked
        // states AFTER consuming a byte, missing state 0.
        if !bytes.is_empty() {
            dfa.mark_possible_future_group(state, *terminal);
        }

        for (index, &byte) in bytes.iter().enumerate() {
            byte_set.insert(byte);

            let prefix = bytes[..=index].to_vec();
            let next_state = if let Some(&existing) = prefixes.get(&prefix) {
                existing
            } else {
                let id = dfa.add_state();
                prefixes.insert(prefix, id);
                id
            };

            if dfa.step(state, byte).is_none() {
                dfa.add_transition(state, byte, next_state);
            }
            state = next_state;

            if index + 1 < bytes.len() {
                dfa.mark_possible_future_group(state, *terminal);
            }
        }

        dfa.mark_finalizer(state, *terminal);
        if *is_non_greedy {
            dfa.mark_non_greedy_finalizer(state, *terminal);
        }
        dfa.set_group_u8set(*terminal, byte_set);
    }

    Tokenizer {
        dfa,
        num_terminals: num_terminals as u32,
    }
}

impl Tokenizer {
    pub fn from_expr_groups(_groups: &[ExprGroup]) -> Self {
        let terminals: Vec<_> = _groups
            .iter()
            .enumerate()
            .flat_map(|(terminal, group)| {
                finite_literals_from_expr(&group.expr, 4096)
                    .into_iter()
                    .flatten()
                    .map(move |bytes| (terminal as TerminalID, bytes, group.is_non_greedy))
            })
            .collect();
        build_literal_tokenizer(&terminals)
    }

    pub fn from_exprs(_terminals: &[(TerminalID, Expr)]) -> Self {
        let terminals: Vec<_> = _terminals
            .iter()
            .flat_map(|(terminal, expr)| {
                finite_literals_from_expr(expr, 4096)
                    .into_iter()
                    .flatten()
                    .map(move |bytes| (*terminal, bytes, false))
            })
            .collect();
        build_literal_tokenizer(&terminals)
    }

    pub fn from_grammar_def(_grammar: &GrammarDef) -> Self {
        let terminals: Vec<_> = _grammar
            .terminals
            .iter()
            .flat_map(|terminal| {
                let expr = parse_regex(_grammar.terminal_pattern(terminal.id));
                finite_literals_from_expr(&expr, 4096)
                    .unwrap_or_else(|| vec![unescape_literal(_grammar.terminal_pattern(terminal.id).as_bytes())])
                    .into_iter()
                    .map(move |bytes| (terminal.id, bytes, false))
            })
            .collect();
        build_literal_tokenizer(&terminals)
    }
}

pub fn parse_regex(_pattern: &str) -> Expr {
    let bytes = _pattern.as_bytes();
    let (expr, pos) = parse_alternation(bytes, 0);
    if pos == bytes.len() {
        expr
    } else {
        Expr::U8Seq(unescape_literal(_pattern.as_bytes()))
    }
}

fn unescape_literal(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        if input[index] == b'\\' && index + 1 < input.len() {
            index += 1;
            out.push(match input[index] {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                other => other,
            });
        } else {
            out.push(input[index]);
        }
        index += 1;
    }
    out
}

fn parse_alternation(_input: &[u8], _pos: usize) -> (Expr, usize) {
    let (mut left, mut pos) = parse_sequence(_input, _pos);
    let mut alts = vec![left];
    while pos < _input.len() && _input[pos] == b'|' {
        let (right, next) = parse_sequence(_input, pos + 1);
        alts.push(right);
        pos = next;
    }
    (if alts.len() == 1 { alts.pop().unwrap() } else { Expr::Choice(alts) }, pos)
}

fn parse_sequence(_input: &[u8], _pos: usize) -> (Expr, usize) {
    let mut parts = Vec::new();
    let mut pos = _pos;
    while pos < _input.len() {
        match _input[pos] {
            b'|' | b')' => break,
            _ => {
                let (expr, next) = parse_quantified(_input, pos);
                parts.push(expr);
                pos = next;
            }
        }
    }
    match parts.len() {
        0 => (Expr::Epsilon, pos),
        1 => (parts.pop().unwrap(), pos),
        _ => (Expr::Seq(parts), pos),
    }
}

fn parse_quantified(_input: &[u8], _pos: usize) -> (Expr, usize) {
    let (mut expr, mut pos) = parse_atom(_input, _pos);
    if pos >= _input.len() {
        return (expr, pos);
    }
    match _input[pos] {
        b'*' => {
            expr = Expr::Repeat { expr: Box::new(expr), min: 0, max: None };
            pos += 1;
        }
        b'+' => {
            expr = Expr::Repeat { expr: Box::new(expr), min: 1, max: None };
            pos += 1;
        }
        b'?' => {
            expr = Expr::Repeat { expr: Box::new(expr), min: 0, max: Some(1) };
            pos += 1;
        }
        b'{' => {
            let (min, max, next) = parse_repetition_bounds(_input, pos + 1);
            expr = Expr::Repeat { expr: Box::new(expr), min, max };
            pos = next;
        }
        _ => {}
    }
    (expr, pos)
}

fn parse_repetition_bounds(_input: &[u8], _pos: usize) -> (usize, Option<usize>, usize) {
    let (min, mut pos) = parse_usize(_input, _pos);
    if pos < _input.len() && _input[pos] == b'}' {
        return (min, Some(min), pos + 1);
    }
    let mut max = None;
    if pos < _input.len() && _input[pos] == b',' {
        pos += 1;
        if pos < _input.len() && _input[pos] != b'}' {
            let (parsed_max, next) = parse_usize(_input, pos);
            max = Some(parsed_max);
            pos = next;
        }
    }
    while pos < _input.len() && _input[pos] != b'}' {
        pos += 1;
    }
    (min, max, pos.saturating_add(1))
}

fn parse_usize(_input: &[u8], _pos: usize) -> (usize, usize) {
    let mut value = 0usize;
    let mut pos = _pos;
    while pos < _input.len() && _input[pos].is_ascii_digit() {
        value = value * 10 + (_input[pos] - b'0') as usize;
        pos += 1;
    }
    (value, pos)
}

fn parse_atom(_input: &[u8], _pos: usize) -> (Expr, usize) {
    if _pos >= _input.len() {
        return (Expr::Epsilon, _pos);
    }
    match _input[_pos] {
        b'(' => {
            let (expr, mut pos) = parse_alternation(_input, _pos + 1);
            if pos < _input.len() && _input[pos] == b')' {
                pos += 1;
            }
            (expr, pos)
        }
        b'[' => parse_char_class(_input, _pos),
        b'\\' => parse_escape(_input, _pos),
        b'.' => (Expr::U8Class(U8Set::all()), _pos + 1),
        byte => (Expr::U8Seq(vec![byte]), _pos + 1),
    }
}

fn parse_char_class(_input: &[u8], _pos: usize) -> (Expr, usize) {
    let mut pos = _pos + 1;
    let mut negate = false;
    if pos < _input.len() && _input[pos] == b'^' {
        negate = true;
        pos += 1;
    }
    let mut set = U8Set::empty();
    while pos < _input.len() && _input[pos] != b']' {
        let start = if _input[pos] == b'\\' {
            let byte = parse_escape_byte(_input, pos);
            pos += escape_len(_input, pos);
            byte
        } else {
            let byte = _input[pos];
            pos += 1;
            byte
        };

        if pos + 1 < _input.len() && _input[pos] == b'-' && _input[pos + 1] != b']' {
            pos += 1;
            let end = if _input[pos] == b'\\' {
                let byte = parse_escape_byte(_input, pos);
                pos += escape_len(_input, pos);
                byte
            } else {
                let byte = _input[pos];
                pos += 1;
                byte
            };
            for byte in start..=end {
                set.insert(byte);
            }
        } else {
            set.insert(start);
        }
    }
    if pos < _input.len() && _input[pos] == b']' {
        pos += 1;
    }
    (Expr::U8Class(if negate { !set } else { set }), pos)
}

fn parse_escape(_input: &[u8], _pos: usize) -> (Expr, usize) {
    if _pos + 1 >= _input.len() {
        return (Expr::U8Seq(vec![b'\\']), _pos + 1);
    }
    let escaped = _input[_pos + 1];
    match escaped {
        b'd' => (Expr::U8Class(U8Set::from_range(b'0', b'9')), _pos + 2),
        b's' => (Expr::U8Class(U8Set::from_bytes(b" \t\r\n")), _pos + 2),
        b'w' => (
            Expr::U8Class(U8Set::from_predicate(|byte| byte.is_ascii_alphanumeric() || byte == b'_')),
            _pos + 2,
        ),
        _ => (Expr::U8Seq(vec![parse_escape_byte(_input, _pos)]), _pos + escape_len(_input, _pos)),
    }
}

fn parse_escape_byte(_input: &[u8], _pos: usize) -> u8 {
    if _pos + 1 >= _input.len() {
        return b'\\';
    }
    match _input[_pos + 1] {
        b'n' => b'\n',
        b'r' => b'\r',
        b't' => b'\t',
        b'x' if _pos + 3 < _input.len() => {
            (hex_digit(_input[_pos + 2]) << 4) | hex_digit(_input[_pos + 3])
        }
        other => other,
    }
}

fn escape_len(_input: &[u8], _pos: usize) -> usize {
    if _pos + 1 < _input.len() && _input[_pos + 1] == b'x' && _pos + 3 < _input.len() {
        4
    } else {
        2
    }
}

fn hex_digit(_b: u8) -> u8 {
    match _b {
        b'0'..=b'9' => _b - b'0',
        b'a'..=b'f' => 10 + (_b - b'a'),
        b'A'..=b'F' => 10 + (_b - b'A'),
        _ => 0,
    }
}
