use crate::glr::grammar::{Production, Symbol};
use json_convertible_derive::JSONConvertible;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
pub struct Item {
    /// Index into the `productions` array passed around the parser generator.
    pub production_id: usize,
    /// Dot position inside `productions[production_id].rhs` (0 <= dot_position <= len).
    pub dot_position: usize,
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
