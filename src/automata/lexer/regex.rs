//! Regex string → `Expr` parsing.
//!
//! This module is concerned only with turning a regex pattern string into an
//! `Expr` AST. Tokenizer construction and expression analysis live under the
//! compiler module (`compiler::compile`).
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::regex::Expr;
use crate::ds::u8set::U8Set;

pub fn parse_regex(_pattern: &str, utf8: bool) -> Expr {
    let bytes = _pattern.as_bytes();
    let (expr, pos) = parse_alternation(bytes, 0, utf8);
    if pos == bytes.len() {
        expr
    } else {
        Expr::U8Seq(unescape_literal(_pattern.as_bytes()))
    }
}

pub(crate) fn unescape_literal(input: &[u8]) -> Vec<u8> {
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

fn parse_alternation(_input: &[u8], _pos: usize, utf8: bool) -> (Expr, usize) {
    let (mut left, mut pos) = parse_sequence(_input, _pos, utf8);
    let mut alts = vec![left];
    while pos < _input.len() && _input[pos] == b'|' {
        let (right, next) = parse_sequence(_input, pos + 1, utf8);
        alts.push(right);
        pos = next;
    }
    (if alts.len() == 1 { alts.pop().unwrap() } else { Expr::Choice(alts) }, pos)
}

fn parse_sequence(_input: &[u8], _pos: usize, utf8: bool) -> (Expr, usize) {
    let mut parts = Vec::new();
    let mut pos = _pos;
    while pos < _input.len() {
        match _input[pos] {
            b'|' | b')' => break,
            _ => {
                let (expr, next) = parse_quantified(_input, pos, utf8);
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

fn parse_quantified(_input: &[u8], _pos: usize, utf8: bool) -> (Expr, usize) {
    let (mut expr, mut pos) = parse_atom(_input, _pos, utf8);
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

fn parse_atom(_input: &[u8], _pos: usize, utf8: bool) -> (Expr, usize) {
    if _pos >= _input.len() {
        return (Expr::Epsilon, _pos);
    }
    match _input[_pos] {
        b'(' => {
            let (expr, mut pos) = parse_alternation(_input, _pos + 1, utf8);
            if pos < _input.len() && _input[pos] == b')' {
                pos += 1;
            }
            (expr, pos)
        }
        b'[' => parse_char_class(_input, _pos, utf8),
        b'\\' => parse_escape(_input, _pos),
        b'.' => (Expr::U8Class(U8Set::all()), _pos + 1),
        byte => (Expr::U8Seq(vec![byte]), _pos + 1),
    }
}

fn parse_char_class(_input: &[u8], _pos: usize, utf8: bool) -> (Expr, usize) {
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
    if negate && utf8 {
        let excluded_is_ascii = set.iter().all(|byte| byte <= 0x7F);
        if excluded_is_ascii {
            return (utf8_aware_negated_ascii_class(set), pos);
        }
    }
    (Expr::U8Class(if negate { !set } else { set }), pos)
}

fn utf8_aware_negated_ascii_class(excluded: U8Set) -> Expr {
    let ascii_allowed = U8Set::from_predicate(|byte| byte <= 0x7F && !excluded.contains(byte));
    let cont = U8Set::from_range(0x80, 0xBF);

    let mut choices = Vec::new();

    if !ascii_allowed.is_empty() {
        choices.push(Expr::U8Class(ascii_allowed));
    }

    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xC2, 0xDF)),
        Expr::U8Class(cont),
    ]));

    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xE0, 0xE0)),
        Expr::U8Class(U8Set::from_range(0xA0, 0xBF)),
        Expr::U8Class(cont),
    ]));
    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xE1, 0xEC)),
        Expr::U8Class(cont),
        Expr::U8Class(cont),
    ]));
    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xED, 0xED)),
        Expr::U8Class(U8Set::from_range(0x80, 0x9F)),
        Expr::U8Class(cont),
    ]));
    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xEE, 0xEF)),
        Expr::U8Class(cont),
        Expr::U8Class(cont),
    ]));

    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xF0, 0xF0)),
        Expr::U8Class(U8Set::from_range(0x90, 0xBF)),
        Expr::U8Class(cont),
        Expr::U8Class(cont),
    ]));
    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xF1, 0xF3)),
        Expr::U8Class(cont),
        Expr::U8Class(cont),
        Expr::U8Class(cont),
    ]));
    choices.push(Expr::Seq(vec![
        Expr::U8Class(U8Set::from_range(0xF4, 0xF4)),
        Expr::U8Class(U8Set::from_range(0x80, 0x8F)),
        Expr::U8Class(cont),
        Expr::U8Class(cont),
    ]));

    Expr::Choice(choices)
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
