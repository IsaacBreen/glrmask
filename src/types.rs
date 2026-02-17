use json_convertible_derive::JSONConvertible;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
pub struct TerminalID(pub usize);
