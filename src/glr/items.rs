use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
// Added for derive macro pattern


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Item {
    pub production: Arc<Production>,
    pub dot_position: usize,
    pub lookahead: Option<Terminal>,
}

// Manual impl for Item (could be derived)
impl JSONConvertible for Item {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("production".to_string(), self.production.to_json());
        obj.insert("dot_position".to_string(), self.dot_position.to_json());
        obj.insert("lookahead".to_string(), self.lookahead.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let production: Production = obj.remove("production").ok_or_else(|| "Missing field production for Item".to_string())
                                    .and_then(Production::from_json)?;
                let dot_position = obj.remove("dot_position").ok_or_else(|| "Missing field dot_position for Item".to_string())
                                      .and_then(usize::from_json)?;
                let lookahead = obj.remove("lookahead").ok_or_else(|| "Missing field lookahead for Item".to_string())
                                      .and_then(Option::<Terminal>::from_json)?;
                Ok(Item { production: Arc::new(production), dot_position, lookahead })
            }
            _ => Err("Expected JSONNode::Object for Item".to_string()),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Display the production and dot position
        write!(f, "[{} ->", self.production.lhs.0)?;
        for (i, symbol) in self.production.rhs.iter().enumerate() {
            if i == self.dot_position {
                write!(f, " •")?;
            }
            match symbol {
                Symbol::Terminal(terminal) => write!(f, " {}", terminal)?,
                Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0)?,
            }
        }
        if self.dot_position == self.production.rhs.len() {
            write!(f, " •")?;
        }
        write!(f, ", ")?;
        // Display the lookahead
        if let Some(lookahead) = &self.lookahead {
            write!(f, "{}", lookahead)?;
        } else {
            write!(f, "ε")?; // Epsilon for no lookahead
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl Item {
    pub fn dot_at_end(&self) -> bool {
        self.dot_position == self.production.rhs.len()
    }

    pub fn next(&self) -> Option<(Symbol, Self)> {
        if let Some(symbol) = self.production.rhs.get(self.dot_position) {
            Some((
                symbol.clone(),
                Item {
                    production: self.production.clone(),
                    dot_position: self.dot_position + 1,
                    lookahead: self.lookahead.clone(),
                },
            ))
        } else {
            None
        }
    }

    pub fn prev(&self) -> Option<(Symbol, Self)> {
        if self.dot_position > 0 {
            let symbol = self.production.rhs[self.dot_position - 1].clone();
            Some((
                symbol,
                Item {
                    production: self.production.clone(),
                    dot_position: self.dot_position - 1,
                    lookahead: self.lookahead.clone(),
                },
            ))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LRMode {
    LALR,
    LALR_EX_GOTO,
    LR1,
}

pub const LR_MODE: LRMode = LRMode::LALR;
// pub const LR_MODE: LRMode = LRMode::LALR_EX_GOTO;
// pub const LR_MODE: LRMode = LRMode::LR1;
