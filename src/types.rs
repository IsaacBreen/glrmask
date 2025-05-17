#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalID(pub usize);

use crate::json_serialization::{JSONNode, JSONConvertible};

impl JSONConvertible for TerminalID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json() // Delegate to usize
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(TerminalID)
    }
}
