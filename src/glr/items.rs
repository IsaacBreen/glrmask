use crate::glr::grammar::{Production, Symbol};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Item {
    /// Index into the `productions` array passed around the parser generator.
    pub production_id: usize,
    /// Dot position inside `productions[production_id].rhs` (0 <= dot_position <= len).
    pub dot_position: usize,
}

impl JSONConvertible for Item {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("production_id".to_string(), self.production_id.to_json());
        obj.insert("dot_position".to_string(), self.dot_position.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let production_id = obj
                    .remove("production_id")
                    .ok_or_else(|| "Missing field production_id for Item".to_string())
                    .and_then(usize::from_json)?;
                let dot_position = obj
                    .remove("dot_position")
                    .ok_or_else(|| "Missing field dot_position for Item".to_string())
                    .and_then(usize::from_json)?;
                Ok(Item {
                    production_id,
                    dot_position,
                })
            }
            _ => Err("Expected JSONNode::Object for Item".to_string()),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[#{} @ {}]", self.production_id, self.dot_position)
    }
}

impl Item {
    /// Returns true if the dot is at the end of the RHS of the corresponding production.
    #[inline]
    pub fn dot_at_end(&self, productions: &[Production]) -> bool {
        self.dot_position == productions[self.production_id].rhs.len()
    }

    /// Returns the symbol *after* the dot and the corresponding next item, if any.
    #[inline]
    pub fn next(&self, productions: &[Production]) -> Option<(Symbol, Self)> {
        let prod = &productions[self.production_id];
        prod.rhs.get(self.dot_position).map(|symbol| {
            (
                symbol.clone(),
                Item {
                    production_id: self.production_id,
                    dot_position: self.dot_position + 1,
                },
            )
        })
    }

    /// Returns the symbol *before* the dot and the corresponding previous item, if any.
    #[inline]
    pub fn prev(&self, productions: &[Production]) -> Option<(Symbol, Self)> {
        if self.dot_position == 0 {
            return None;
        }
        let prod = &productions[self.production_id];
        let symbol = prod.rhs[self.dot_position - 1].clone();
        Some((
            symbol,
            Item {
                production_id: self.production_id,
                dot_position: self.dot_position - 1,
            },
        ))
    }
}
