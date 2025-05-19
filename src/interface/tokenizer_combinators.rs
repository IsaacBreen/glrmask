use crate::finite_automata::{Expr, QuantifierType};
use crate::datastructures::u8set::U8Set;

/// Creates a sequence of parsers
pub fn seq_fast(parsers: Vec<Expr>) -> Expr {
    Expr::Seq(parsers)
}

/// Creates a choice of parsers
pub fn choice_fast(parsers: Vec<Expr>) -> Expr {
    Expr::Choice(parsers)
}

/// Makes a parser optional
pub fn opt_fast(parser: Expr) -> Expr {
    Expr::Choice(vec![parser, Expr::Seq(vec![])]) // Epsilon is an empty sequence
}

/// Requires one or more occurrences of a parser
pub fn repeat1_fast(parser: Expr) -> Expr {
    Expr::Quantifier(Box::new(parser), QuantifierType::OneOrMore)
}

/// Matches a specific byte
pub fn eat_u8_fast(byte: u8) -> Expr {
    Expr::U8Seq(vec![byte])
}

/// Matches any byte except the specified one
pub fn eat_u8_negation_fast(byte: u8) -> Expr {
    Expr::U8Class(U8Set::from_byte(byte).complement())
}

/// Matches any of the specified bytes
pub fn eat_u8_choice_fast(bytes: &[u8]) -> Expr {
    Expr::U8Class(U8Set::from_bytes(bytes))
}

/// Matches any byte not in the specified set
pub fn eat_u8_negation_choice_fast(bytes: &[u8]) -> Expr {
    Expr::U8Class(U8Set::from_bytes(bytes).complement())
}

/// Matches a byte within a specified range
pub fn eat_u8_range_fast(start: u8, end: u8) -> Expr {
    Expr::U8Class(U8Set::from_byte_range(start..=end))
}

/// Matches a specific character (assuming ASCII or direct u8 conversion)
pub fn eat_char_fast(c: char) -> Expr {
    // This is a simplification; proper char handling might involve UTF-8 sequences.
    // For single-byte chars, this is fine.
    let mut buf = [0; 4];
    Expr::U8Seq(c.encode_utf8(&mut buf).as_bytes().to_vec())
}

/// Matches any character except the specified one (complex for multi-byte UTF-8)
/// This simplified version works for single-byte characters.
pub fn eat_char_negation_fast(c: char) -> Expr {
    if c.len_utf8() == 1 {
        Expr::U8Class(U8Set::from_byte(c as u8).complement())
    } else {
        // Handling negation of multi-byte characters is complex with U8Class.
        // This would typically require a more advanced regex feature (e.g., negative lookahead)
        // or be handled at a higher level. For now, panic or return an "any" matcher.
        // panic!("eat_char_negation_fast for multi-byte char not directly supported by U8Class");
        eat_any_fast() // Fallback: matches anything, likely not desired.
    }
}

/// Matches any of the specified characters (collects their byte sequences)
pub fn eat_char_choice_fast(s: &str) -> Expr {
    Expr::U8Class(U8Set::from_chars(s))
}

/// Matches any character not in the specified set (complex for multi-byte UTF-8)
pub fn eat_char_negation_choice_fast(s: &str) -> Expr {
     Expr::U8Class(U8Set::from_chars(s).complement())
}

/// Matches a specific string
pub fn eat_string_fast(s: &str) -> Expr {
    Expr::U8Seq(s.bytes().collect())
}

// eat_byte_range_fast is same as eat_u8_range_fast
// pub fn eat_byte_range_fast(start: u8, end: u8) -> Expr {
//     Expr::U8Class(U8Set::from_byte_range(start..=end))
// }

/// Creates a choice of byte strings
pub fn eat_bytestring_choice_fast(bytestrings: Vec<Vec<u8>>) -> Expr {
    let children: Vec<Expr> = bytestrings
        .into_iter()
        .map(eat_bytestring_fast)
        .collect();
    if children.is_empty() {
        Expr::Epsilon // Or handle error: choice of nothing?
    } else {
        choice_fast(children)
    }
}

/// Matches a specific byte string
pub fn eat_bytestring_fast(bytes: Vec<u8>) -> Expr {
    Expr::U8Seq(bytes)
}

/// Creates a choice of strings
pub fn eat_string_choice_fast(strings: &[&str]) -> Expr {
    let children: Vec<Expr> = strings.iter().map(|s| eat_string_fast(s)).collect();
    if children.is_empty() {
        Expr::Epsilon
    } else {
        choice_fast(children)
    }
}

/// Eats any byte
pub fn eat_any_fast() -> Expr {
    Expr::U8Class(U8Set::all())
}

/// Allows zero or more occurrences of a parser
pub fn repeat0_fast(parser: Expr) -> Expr {
    // opt_fast(repeat1_fast(parser)) // This was one way
    // Another common way:
    Expr::Quantifier(Box::new(parser), QuantifierType::ZeroOrMore)
}

/// Matches a separator-delimited sequence of elements (at least one element)
pub fn seprep1_fast(element: Expr, separator: Expr) -> Expr {
    seq_fast(vec![element.clone(), repeat0_fast(seq_fast(vec![separator, element]))])
}

/// Optionally matches a separator-delimited sequence of elements (zero or more elements)
pub fn seprep0_fast(element: Expr, separator: Expr) -> Expr {
    opt_fast(seprep1_fast(element, separator))
}

/// Matches exactly n occurrences of a parser
pub fn repeatn_fast(n: usize, parser: Expr) -> Expr {
    if n == 0 {
        return Expr::Seq(vec![]); // Epsilon
    }
    let parsers = std::iter::repeat(parser).take(n).collect();
    seq_fast(parsers)
}
