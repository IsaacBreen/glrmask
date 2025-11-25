use json_convertible_derive::JSONConvertible;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct TerminalID(pub usize);
