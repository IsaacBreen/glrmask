//! Regex string → `Expr` parsing.
//!
//! This module is concerned only with turning a regex pattern string into an
//! `Expr` AST. Tokenizer construction and expression analysis live under the
//! compiler module (`compiler::compile`).

use regex_syntax::utf8::Utf8Sequences;

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
    U8Set::from_bytes(b" \t\r\n\x0B\x0C")
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

fn parse_unicode_hex(input: &[u8], start: usize, digits: usize) -> Option<(u32, usize)> {
    let end = start.checked_add(digits)?;
    let mut value = 0u32;
    for &byte in input.get(start..end)? {
        let digit = match byte {
            b'0'..=b'9' => u32::from(byte - b'0'),
            b'a'..=b'f' => 10 + u32::from(byte - b'a'),
            b'A'..=b'F' => 10 + u32::from(byte - b'A'),
            _ => return None,
        };
        value = value.checked_mul(16)?.checked_add(digit)?;
    }
    Some((value, end))
}

/// Decode a standard `\\uXXXX` or `\\UXXXXXXXX` escape at `pos`.
/// Adjacent UTF-16 high/low surrogate escapes are combined into one scalar.
pub fn decode_unicode_escape(input: &[u8], pos: usize) -> Option<(char, usize)> {
    if input.get(pos) != Some(&b'\\') {
        return None;
    }
    let (value, next, short) = match input.get(pos + 1).copied()? {
        b'u' => {
            let (value, next) = parse_unicode_hex(input, pos + 2, 4)?;
            (value, next, true)
        }
        b'U' => {
            let (value, next) = parse_unicode_hex(input, pos + 2, 8)?;
            (value, next, false)
        }
        _ => return None,
    };
    if short && (0xD800..=0xDBFF).contains(&value) {
        if input.get(next..next + 2)? != b"\\u" {
            return None;
        }
        let (low, end) = parse_unicode_hex(input, next + 2, 4)?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return None;
        }
        let scalar = 0x1_0000 + ((value - 0xD800) << 10) + (low - 0xDC00);
        return char::from_u32(scalar).map(|character| (character, end));
    }
    if (0xD800..=0xDFFF).contains(&value) {
        return None;
    }
    char::from_u32(value).map(|character| (character, next))
}

fn parse_atom(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    if pos >= input.len() {
        return (Expr::Epsilon, pos);
    }
    match input[pos] {
        b'(' => parse_group(input, pos, utf8),
        b'[' => parse_char_class(input, pos, utf8),
        b'\\' => parse_escape(input, pos, utf8),
        b'.' => {
            let expr = if utf8 {
                utf8_aware_negated_ascii_class(U8Set::empty())
            } else {
                Expr::U8Class(U8Set::all())
            };
            (expr, pos + 1)
        }
        byte if utf8 && !byte.is_ascii() => {
            let next = utf8_char_end(input, pos);
            (Expr::U8Seq(input[pos..next].to_vec()), next)
        }
        byte => (Expr::U8Seq(vec![byte]), pos + 1),
    }
}

fn parse_group(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    let inner_pos = consume_group_prefix(input, pos + 1);
    let (expr, mut pos) = parse_alternation(input, inner_pos, utf8);
    if pos < input.len() && input[pos] == b')' {
        pos += 1;
    }
    (expr, pos)
}

fn consume_group_prefix(input: &[u8], pos: usize) -> usize {
    if pos >= input.len() || input[pos] != b'?' {
        return pos;
    }

    if pos + 1 < input.len() && input[pos + 1] == b':' {
        return pos + 2;
    }

    if pos + 2 < input.len() && input[pos + 1] == b'P' && input[pos + 2] == b'<' {
        return consume_named_group_name(input, pos + 3).unwrap_or(pos);
    }

    if pos + 1 < input.len() && input[pos + 1] == b'<' {
        return consume_named_group_name(input, pos + 2).unwrap_or(pos);
    }

    pos
}

fn consume_named_group_name(input: &[u8], mut pos: usize) -> Option<usize> {
    while pos < input.len() {
        match input[pos] {
            b'>' => return Some(pos + 1),
            b')' => return None,
            _ => pos += 1,
        }
    }

    None
}

fn utf8_char_end(input: &[u8], pos: usize) -> usize {
    let remaining = std::str::from_utf8(&input[pos..])
        .expect("regex pattern originated from valid UTF-8");
    pos + remaining
        .chars()
        .next()
        .expect("position must contain a UTF-8 character")
        .len_utf8()
}

fn utf8_char_at(input: &[u8], pos: usize) -> (char, usize) {
    let remaining = std::str::from_utf8(&input[pos..])
        .expect("regex pattern originated from valid UTF-8");
    let character = remaining
        .chars()
        .next()
        .expect("position must contain a UTF-8 character");
    (character, pos + character.len_utf8())
}

fn utf8_sequence_expr(sequence: &regex_syntax::utf8::Utf8Sequence) -> Expr {
    sequence_or_single(
        sequence
            .as_slice()
            .iter()
            .map(|range| Expr::U8Class(U8Set::from_range(range.start, range.end)))
            .collect(),
    )
}

fn utf8_range_expr(start: char, end: char) -> Expr {
    choice_or_single(
        Utf8Sequences::new(start, end)
            .map(|sequence| utf8_sequence_expr(&sequence))
            .collect(),
    )
}

fn char_class_contains_direct_unicode(input: &[u8], pos: usize) -> bool {
    let mut cursor = pos + 1;
    if cursor < input.len() && input[cursor] == b'^' {
        cursor += 1;
    }
    while cursor < input.len() && input[cursor] != b']' {
        if input[cursor] == b'\\' {
            if decode_unicode_escape(input, cursor).is_some() {
                return true;
            }
            cursor += 1;
            if cursor >= input.len() {
                break;
            }
        }
        if !input[cursor].is_ascii() {
            return true;
        }
        cursor += 1;
    }
    false
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

fn parse_byte_char_class(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
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

#[derive(Clone, Copy)]
enum Utf8ClassAtom {
    Byte(u8),
    Unicode(char),
}

fn parse_utf8_class_atom(input: &[u8], pos: usize) -> Option<(Utf8ClassAtom, usize)> {
    if pos >= input.len() || input[pos] == b']' {
        return None;
    }
    if input[pos] == b'\\' {
        if let Some((character, next)) = decode_unicode_escape(input, pos) {
            return Some((Utf8ClassAtom::Unicode(character), next));
        }
        if pos + 1 >= input.len() {
            return Some((Utf8ClassAtom::Byte(b'\\'), pos + 1));
        }
        if !input[pos + 1].is_ascii() {
            let (character, next) = utf8_char_at(input, pos + 1);
            return Some((Utf8ClassAtom::Unicode(character), next));
        }
        return Some((
            Utf8ClassAtom::Byte(parse_escape_byte(input, pos)),
            pos + escape_len(input, pos),
        ));
    }
    if input[pos].is_ascii() {
        Some((Utf8ClassAtom::Byte(input[pos]), pos + 1))
    } else {
        let (character, next) = utf8_char_at(input, pos);
        Some((Utf8ClassAtom::Unicode(character), next))
    }
}

fn add_utf8_class_atom(
    atom: Utf8ClassAtom,
    byte_set: &mut U8Set,
    unicode_ranges: &mut Vec<(char, char)>,
) {
    match atom {
        Utf8ClassAtom::Byte(byte) => {
            byte_set.insert(byte);
        }
        Utf8ClassAtom::Unicode(character) => unicode_ranges.push((character, character)),
    }
}

fn add_utf8_class_range(
    start: Utf8ClassAtom,
    end: Utf8ClassAtom,
    byte_set: &mut U8Set,
    unicode_ranges: &mut Vec<(char, char)>,
) {
    match (start, end) {
        (Utf8ClassAtom::Byte(start), Utf8ClassAtom::Byte(end)) => {
            for byte in start..=end {
                byte_set.insert(byte);
            }
        }
        (Utf8ClassAtom::Unicode(start), Utf8ClassAtom::Unicode(end)) if start <= end => {
            unicode_ranges.push((start, end));
        }
        (Utf8ClassAtom::Byte(start), Utf8ClassAtom::Unicode(end))
            if start.is_ascii() && (start as char) <= end =>
        {
            unicode_ranges.push((start as char, end));
        }
        (Utf8ClassAtom::Unicode(start), Utf8ClassAtom::Byte(end))
            if end.is_ascii() && start <= end as char =>
        {
            unicode_ranges.push((start, end as char));
        }
        (start, end) => {
            add_utf8_class_atom(start, byte_set, unicode_ranges);
            byte_set.insert(b'-');
            add_utf8_class_atom(end, byte_set, unicode_ranges);
        }
    }
}

fn positive_utf8_class_expr(byte_set: U8Set, unicode_ranges: Vec<(char, char)>) -> Expr {
    let mut options = Vec::new();
    if !byte_set.is_empty() {
        options.push(Expr::U8Class(byte_set));
    }
    options.extend(
        unicode_ranges
            .into_iter()
            .map(|(start, end)| utf8_range_expr(start, end)),
    );
    choice_or_single(options)
}

fn parse_unicode_char_class(input: &[u8], pos: usize) -> (Expr, usize) {
    let mut cursor = pos + 1;
    let mut negate = false;
    if cursor < input.len() && input[cursor] == b'^' {
        negate = true;
        cursor += 1;
    }

    let mut byte_set = U8Set::empty();
    let mut unicode_ranges = Vec::new();
    while cursor < input.len() && input[cursor] != b']' {
        if input[cursor] == b'\\' {
            if let Some((escape_set, next)) = parse_escape_class_set(input, cursor) {
                byte_set = byte_set.union(&escape_set);
                cursor = next;
                continue;
            }
        }

        let Some((start, next)) = parse_utf8_class_atom(input, cursor) else {
            break;
        };
        cursor = next;
        if cursor + 1 < input.len()
            && input[cursor] == b'-'
            && input[cursor + 1] != b']'
        {
            if let Some((end, next)) = parse_utf8_class_atom(input, cursor + 1) {
                add_utf8_class_range(start, end, &mut byte_set, &mut unicode_ranges);
                cursor = next;
                continue;
            }
        }
        add_utf8_class_atom(start, &mut byte_set, &mut unicode_ranges);
    }
    if cursor < input.len() && input[cursor] == b']' {
        cursor += 1;
    }

    let excluded_or_allowed = positive_utf8_class_expr(byte_set, unicode_ranges);
    if negate {
        (
            Expr::Exclude {
                expr: Box::new(utf8_aware_negated_ascii_class(U8Set::empty())),
                exclude: Box::new(excluded_or_allowed),
            },
            cursor,
        )
    } else {
        (excluded_or_allowed, cursor)
    }
}

fn parse_char_class(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {
    if utf8 && char_class_contains_direct_unicode(input, pos) {
        parse_unicode_char_class(input, pos)
    } else {
        parse_byte_char_class(input, pos, utf8)
    }
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
    if utf8 && let Some((character, next)) = decode_unicode_escape(input, pos) {
        let mut encoded = [0u8; 4];
        return (
            Expr::U8Seq(character.encode_utf8(&mut encoded).as_bytes().to_vec()),
            next,
        );
    }
    let escaped = input[pos + 1];
    if utf8 && !escaped.is_ascii() {
        let next = utf8_char_end(input, pos + 1);
        return (Expr::U8Seq(input[pos + 1..next].to_vec()), next);
    }
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



#[cfg(test)]
mod tests {
    use super::*;

    fn expression_matches(expr: Expr, input: &[u8]) -> bool {
        use std::sync::Arc;

        use crate::automata::lexer::tokenizer::Lexer as _;

        let tokenizer = expr.clone().build().into_tokenizer(1, Some(Arc::from([expr])));
        tokenizer
            .execute_from_state(input, tokenizer.initial_state())
            .matches
            .iter()
            .any(|matched| matched.id == 0 && matched.width == input.len())
    }

    #[test]
    fn direct_unicode_atom_quantifier_applies_to_the_whole_scalar() {
        assert_eq!(
            parse_regex("—+", true),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq("—".as_bytes().to_vec())),
                min: 1,
                max: None,
            }
        );
    }

    #[test]
    fn direct_unicode_classes_lower_to_utf8_sequences() {
        let singleton = parse_regex("[—]", true);
        assert_eq!(
            singleton,
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_range(0xE2, 0xE2)),
                Expr::U8Class(U8Set::from_range(0x80, 0x80)),
                Expr::U8Class(U8Set::from_range(0x94, 0x94)),
            ])
        );

        let range = parse_regex("[é-ê]", true);
        assert_eq!(
            range,
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_range(0xC3, 0xC3)),
                Expr::U8Class(U8Set::from_range(0xA9, 0xAA)),
            ])
        );

        let byte_escape = parse_regex(r"[\xE2]", true);
        assert_eq!(byte_escape, Expr::U8Class(U8Set::from_range(0xE2, 0xE2)));
    }

    #[test]
    fn direct_unicode_singletons_cover_utf8_scalar_boundaries() {
        let scalars = [
            '\u{80}',
            '\u{7ff}',
            '\u{800}',
            '\u{d7ff}',
            '\u{e000}',
            '\u{ffff}',
            '\u{10000}',
            '\u{10ffff}',
            'é',
            '—',
            '😀',
        ];
        for scalar in scalars {
            let pattern = format!("[{scalar}]");
            let expr = parse_regex(&pattern, true);
            let mut encoded = [0u8; 4];
            let exact = scalar.encode_utf8(&mut encoded).as_bytes();
            assert!(expression_matches(expr.clone(), exact), "{pattern}");
            for prefix_len in 0..exact.len() {
                assert!(
                    !expression_matches(expr.clone(), &exact[..prefix_len]),
                    "{pattern} accepted prefix length {prefix_len}"
                );
            }
            assert!(!expression_matches(expr, b"a"), "{pattern}");
        }
    }

    #[test]
    fn direct_unicode_ranges_and_negation_match_scalar_semantics() {
        let cases = [
            ("[é-ê]", vec!['é', 'ê'], vec!['è', 'ë', 'a']),
            ("[\u{7ff}-\u{800}]", vec!['\u{7ff}', '\u{800}'], vec!['\u{7fe}', '\u{801}']),
            ("[\u{d7ff}-\u{e000}]", vec!['\u{d7ff}', '\u{e000}'], vec!['a']),
            ("[\u{ffff}-\u{10000}]", vec!['\u{ffff}', '\u{10000}'], vec!['\u{fffe}', '\u{10001}']),
            ("[😀-😃]", vec!['😀', '😁', '😂', '😃'], vec!['😄', 'a']),
        ];
        for (pattern, accepted, rejected) in cases {
            let expr = parse_regex(pattern, true);
            for scalar in accepted {
                let mut encoded = [0u8; 4];
                assert!(
                    expression_matches(expr.clone(), scalar.encode_utf8(&mut encoded).as_bytes()),
                    "{pattern} rejected {scalar:?}"
                );
            }
            for scalar in rejected {
                let mut encoded = [0u8; 4];
                assert!(
                    !expression_matches(expr.clone(), scalar.encode_utf8(&mut encoded).as_bytes()),
                    "{pattern} accepted {scalar:?}"
                );
            }
        }

        let negated = parse_regex("[^—😀]", true);
        for accepted in ['a', 'é', '😃'] {
            let mut encoded = [0u8; 4];
            assert!(expression_matches(
                negated.clone(),
                accepted.encode_utf8(&mut encoded).as_bytes()
            ));
        }
        for rejected in ['—', '😀'] {
            let mut encoded = [0u8; 4];
            assert!(!expression_matches(
                negated.clone(),
                rejected.encode_utf8(&mut encoded).as_bytes()
            ));
        }
    }

    #[test]
    fn mixed_byte_escapes_and_direct_unicode_remain_distinct() {
        let mixed = parse_regex(r"[A-Z\xE2—]", true);
        for accepted in [b"A".as_slice(), b"Z".as_slice(), &[0xE2], "—".as_bytes()] {
            assert!(expression_matches(mixed.clone(), accepted), "{accepted:?}");
        }
        for rejected in [b"a".as_slice(), &[0x80], "–".as_bytes()] {
            assert!(!expression_matches(mixed.clone(), rejected), "{rejected:?}");
        }
    }


    #[test]
    fn standard_unicode_escape_ast_is_one_scalar() {
        assert_eq!(
            parse_regex(r" \u2014", true),
            Expr::Seq(vec![
                Expr::U8Seq(vec![b' ']),
                Expr::U8Seq("—".as_bytes().to_vec()),
            ])
        );
        assert_eq!(
            parse_regex(r"[\u2014]", true),
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_range(0xE2, 0xE2)),
                Expr::U8Class(U8Set::from_range(0x80, 0x80)),
                Expr::U8Class(U8Set::from_range(0x94, 0x94)),
            ])
        );
    }

}
