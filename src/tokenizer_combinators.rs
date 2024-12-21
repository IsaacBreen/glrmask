use crate::finite_automata::{Expr, QuantifierType};
use crate::u8set::U8Set;

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
    Expr::Choice(vec![parser, Expr::Seq(vec![])])
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

/// Matches a specific character
pub fn eat_char_fast(c: char) -> Expr {
    Expr::U8Seq(vec![c as u8])
}

/// Matches any character except the specified one
pub fn eat_char_negation_fast(c: char) -> Expr {
    Expr::U8Class(U8Set::from_char(c).complement())
}

/// Matches any of the specified characters
pub fn eat_char_choice_fast(s: &str) -> Expr {
    Expr::U8Class(U8Set::from_chars(s))
}

/// Matches any character not in the specified set
pub fn eat_char_negation_choice_fast(s: &str) -> Expr {
    Expr::U8Class(U8Set::from_chars(s).complement())
}

/// Matches a specific string
pub fn eat_string_fast(s: &str) -> Expr {
    Expr::U8Seq(s.bytes().collect())
}

/// Matches a byte within a specified range
pub fn eat_byte_range_fast(start: u8, end: u8) -> Expr {
    Expr::U8Class(U8Set::from_byte_range(start..=end))
}

/// Creates a choice of byte strings
pub fn eat_bytestring_choice_fast(bytestrings: Vec<Vec<u8>>) -> Expr {
    let children: Vec<Expr> = bytestrings
        .into_iter()
        .map(eat_bytestring_fast)
        .collect();
    choice_fast(children)
}

/// Matches a specific byte string
pub fn eat_bytestring_fast(bytes: Vec<u8>) -> Expr {
    Expr::U8Seq(bytes)
}

/// Creates a choice of strings
pub fn eat_string_choice_fast(strings: &[&str]) -> Expr {
    choice_fast(strings.iter().map(|s| eat_string_fast(s)).collect())
}

/// Allows zero or more occurrences of a parser
pub fn repeat0_fast(parser: Expr) -> Expr {
    opt_fast(repeat1_fast(parser))
}

/// Matches a separator-delimited sequence of elements
pub fn seprep1_fast(a: Expr, b: Expr) -> Expr {
    seq_fast(vec![a.clone(), repeat0_fast(seq_fast(vec![b, a]))])
}

/// Optionally matches a separator-delimited sequence of elements
pub fn seprep0_fast(a: Expr, b: Expr) -> Expr {
    opt_fast(seprep1_fast(a, b))
}

/// Matches exactly n occurrences of a parser
pub fn repeatn_fast(n: usize, parser: Expr) -> Expr {
    if n == 0 {
        return seq_fast(vec![]);
    }
    let parsers = std::iter::repeat(parser).take(n).collect();
    seq_fast(parsers)
}

/// Macro for creating a sequence of parsers
#[macro_export]
macro_rules! seq_fast {
    ($($x:expr),* $(,)?) => {
        $crate::tokenizer_combinators::seq_fast(vec![$($x),*])
    };
}

/// Macro for creating a choice of parsers
#[macro_export]
macro_rules! choice_fast {
    ($($x:expr),* $(,)?) => {
        $crate::tokenizer_combinators::choice_fast(vec![$($x),*])
    };
}
