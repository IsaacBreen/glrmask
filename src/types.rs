use crate::json_serialization::{JSONConvertible, JSONNode};
// Added for derive macro pattern

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalID(pub usize);

// Manual impl for TerminalID (could be derived)
impl JSONConvertible for TerminalID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json() // Delegate to usize's implementation
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(TerminalID)
    }
}
