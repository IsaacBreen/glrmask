//! Regex string → `Expr` parsing.
//!
//! This module is concerned only with turning a regex pattern string into an
//! `Expr` AST. Tokenizer construction and expression analysis live under the
//! compiler module (`compiler::compile`).

use crate::automata::regex::Expr;
use crate::ds::u8set::U8Set;

fn choice_or_single(mut options: Vec<Expr>) -> Expr {
    if options.len() == 1 {
        options.pop().unwrap()
    } else {
        Expr::Choice(options)
    }
}

fn sequence_or_single(mut parts: Vec<Expr>) -> Expr {
    match parts.len() {
        0 => Expr::Epsilon,
        1 => parts.pop().unwrap(),
        _ => Expr::Seq(parts),
    }
}

fn repeat_expr(expr: Expr, min: usize, max: Option<usize>) -> Expr {
    Expr::Repeat {
        expr: Box::new(expr),
        min,
        max,
    }
}

fn ascii_digit_set() -> U8Set {
    U8Set::from_range(b'0', b'9')
}

fn ascii_space_set() -> U8Set {
    U8Set::from_bytes(b" \t\r\n")
}

fn ascii_word_set() -> U8Set {
    U8Set::from_predicate(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn escaped_class_set(escaped: u8) -> Option<U8Set> {
    match escaped {
        b'd' => Some(ascii_digit_set()),
        b's' => Some(ascii_space_set()),
        b'w' => Some(ascii_word_set()),
        _ => None,
    }
}

pub fn parse_regex(pattern: &str, utf8: bool) -> Expr {
    let bytes = pattern.as_bytes();
    let (expr, pos) = parse_alternation(bytes, 0, utf8);
    if pos == bytes.len() {
        expr
    } else {
        Expr::U8Seq(unescape_literal(pattern.as_bytes()))
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

fn parse_alternation(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    let (left, mut pos) = parse_sequence(input, pos, utf8);
    let mut alts = vec![left];
    while pos < input.len() && input[pos] == b'|' {
        let (right, next) = parse_sequence(input, pos + 1, utf8);
        alts.push(right);
        pos = next;
    }
    (choice_or_single(alts), pos)
}

fn parse_sequence(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    let mut parts = Vec::new();
    let mut pos = pos;
    while pos < input.len() {
        match input[pos] {
            b'|' | b')' => break,
            _ => {
                let (expr, next) = parse_quantified(input, pos, utf8);
                parts.push(expr);
                pos = next;
            }
        }
    }
    (sequence_or_single(parts), pos)
}

fn parse_quantified(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    let (mut expr, mut pos) = parse_atom(input, pos, utf8);
    if pos >= input.len() {
        return (expr, pos);
    }
    match input[pos] {
        b'*' => {
            expr = repeat_expr(expr, 0, None);
            pos += 1;
            pos = consume_lazy_suffix(input, pos);
        }
        b'+' => {
            expr = repeat_expr(expr, 1, None);
            pos += 1;
            pos = consume_lazy_suffix(input, pos);
        }
        b'?' => {
            expr = repeat_expr(expr, 0, Some(1));
            pos += 1;
            pos = consume_lazy_suffix(input, pos);
        }
        b'{' => {
            let (min, max, next) = parse_repetition_bounds(input, pos + 1);
            expr = repeat_expr(expr, min, max);
            pos = next;
            pos = consume_lazy_suffix(input, pos);
        }
        _ => {}
    }
    (expr, pos)
}

fn consume_lazy_suffix(input: &[u8], pos: usize) -> usize {
    if pos < input.len() && input[pos] == b'?' {
        pos + 1
    } else {
        pos
    }
}

fn parse_repetition_bounds(input: &[u8], pos: usize) -> (usize, Option<usize>, usize) {
    let (min, mut pos) = parse_usize(input, pos);
    if pos < input.len() && input[pos] == b'}' {
        return (min, Some(min), pos + 1);
    }
    let mut max = None;
    if pos < input.len() && input[pos] == b',' {
        pos += 1;
        if pos < input.len() && input[pos] != b'}' {
            let (parsed_max, next) = parse_usize(input, pos);
            max = Some(parsed_max);
            pos = next;
        }
    }
    while pos < input.len() && input[pos] != b'}' {
        pos += 1;
    }
    (min, max, pos.saturating_add(1))
}

fn parse_usize(input: &[u8], pos: usize) -> (usize, usize) {
    let mut value = 0usize;
    let mut pos = pos;
    while pos < input.len() && input[pos].is_ascii_digit() {
        value = value * 10 + (input[pos] - b'0') as usize;
        pos += 1;
    }
    (value, pos)
}

fn parse_atom(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    if pos >= input.len() {
        return (Expr::Epsilon, pos);
    }
    match input[pos] {
        b'(' => parse_group(input, pos, utf8),
        b'[' => parse_char_class(input, pos, utf8),
        b'\\' => parse_escape(input, pos, utf8),
        b'.' => (Expr::U8Class(U8Set::all()), pos + 1),
        byte => (Expr::U8Seq(vec![byte]), pos + 1),
    }
}

fn parse_group(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    let inner_pos = consume_non_capturing_group_prefix(input, pos + 1);
    let (expr, mut pos) = parse_alternation(input, inner_pos, utf8);
    if pos < input.len() && input[pos] == b')' {
        pos += 1;
    }
    (expr, pos)
}

fn consume_non_capturing_group_prefix(input: &[u8], pos: usize) -> usize {
    if pos + 1 < input.len() && input[pos] == b'?' && input[pos + 1] == b':' {
        pos + 2
    } else {
        pos
    }
}

fn parse_char_class_byte(input: &[u8], pos: usize) -> Option<(u8, usize)> {
    if pos >= input.len() {
        return None;
    }

    if input[pos] == b'\\' {
        Some((parse_escape_byte(input, pos), pos + escape_len(input, pos)))
    } else {
        Some((input[pos], pos + 1))
    }
}

fn parse_char_class(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    let mut pos = pos + 1;
    let mut negate = false;
    if pos < input.len() && input[pos] == b'^' {
        negate = true;
        pos += 1;
    }
    let mut set = U8Set::empty();
    while pos < input.len() && input[pos] != b']' {
        if input[pos] == b'\\' {
            if let Some((escape_set, next_pos)) = parse_escape_class_set(input, pos) {
                set = set.union(&escape_set);
                pos = next_pos;
                continue;
            }
        }

        let Some((start, next_pos)) = parse_char_class_byte(input, pos) else {
            break;
        };
        pos = next_pos;

        if pos + 1 < input.len() && input[pos] == b'-' && input[pos + 1] != b']' {
            let Some((end, next_pos)) = parse_char_class_byte(input, pos + 1) else {
                break;
            };
            pos = next_pos;
            for byte in start..=end {
                set.insert(byte);
            }
        } else {
            set.insert(start);
        }
    }
    if pos < input.len() && input[pos] == b']' {
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

fn parse_escape_class_set(input: &[u8], pos: usize) -> Option<(U8Set, usize)> {
    if pos + 1 >= input.len() {
        return None;
    }
    let set = escaped_class_set(input[pos + 1])?;
    Some((set, pos + 2))
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

fn parse_escape(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    if pos + 1 >= input.len() {
        return (Expr::U8Seq(vec![b'\\']), pos + 1);
    }
    let escaped = input[pos + 1];
    match escaped {
        b'd' => (Expr::U8Class(ascii_digit_set()), pos + 2),
        b's' => (Expr::U8Class(ascii_space_set()), pos + 2),
        b'w' => (Expr::U8Class(ascii_word_set()), pos + 2),
        b'D' => (negated_ascii_class(ascii_digit_set(), utf8), pos + 2),
        b'S' => (negated_ascii_class(ascii_space_set(), utf8), pos + 2),
        b'W' => (negated_ascii_class(ascii_word_set(), utf8), pos + 2),
        _ => (Expr::U8Seq(vec![parse_escape_byte(input, pos)]), pos + escape_len(input, pos)),
    }
}

fn negated_ascii_class(excluded: U8Set, utf8: bool) -> Expr {
    if utf8 {
        utf8_aware_negated_ascii_class(excluded)
    } else {
        Expr::U8Class(!excluded)
    }
}

fn parse_escape_byte(input: &[u8], pos: usize) -> u8 {
    if pos + 1 >= input.len() {
        return b'\\';
    }
    match input[pos + 1] {
        b'n' => b'\n',
        b'r' => b'\r',
        b't' => b'\t',
        b'x' if pos + 3 < input.len() => {
            (hex_digit(input[pos + 2]) << 4) | hex_digit(input[pos + 3])
        }
        other => other,
    }
}

fn escape_len(input: &[u8], pos: usize) -> usize {
    if pos + 1 < input.len() && input[pos + 1] == b'x' && pos + 3 < input.len() {
        4
    } else {
        2
    }
}

fn hex_digit(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => 10 + (b - b'a'),
        b'A'..=b'F' => 10 + (b - b'A'),
        _ => 0,
    }
}
