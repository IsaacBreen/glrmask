use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap;
use std::fmt::{Display, Formatter};
// Added for derive macro pattern


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Item {
    pub production: Production,
    pub dot_position: usize,
    pub lookahead: Terminal,
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
                let production = obj.remove("production").ok_or_else(|| "Missing field production for Item".to_string())
                                    .and_then(Production::from_json)?;
                let dot_position = obj.remove("dot_position").ok_or_else(|| "Missing field dot_position for Item".to_string())
                                      .and_then(usize::from_json)?;
                let lookahead = obj.remove("lookahead").ok_or_else(|| "Missing field lookahead for Item".to_string())
                                     .and_then(Terminal::from_json)?;
                Ok(Item { production, dot_position, lookahead })
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
        write!(f, ", {}]", self.lookahead)
    }
}

pub fn compute_closure(
    items: &BTreeSet<Item>,
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_set: &BTreeSet<NonTerminal>,
) -> BTreeSet<Item> {
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        // If the dot is at the end of a production, do nothing.
        if item.dot_position >= item.production.rhs.len() {
            continue;
        }

        // Get the symbol after the dot, which must be a non-terminal B for closure to expand.
        if let Some(Symbol::NonTerminal(b)) = item.production.rhs.get(item.dot_position) {
            // This is an item of the form [A -> α . B β, a]
            // We need to compute FIRST(βa)
            let beta = &item.production.rhs[item.dot_position + 1..];
            let a = &item.lookahead;

            // Compute FIRST of β
            let mut lookaheads = BTreeSet::new();
            let mut beta_is_nullable = true;

            for symbol in beta {
                match symbol {
                    Symbol::Terminal(t) => {
                        lookaheads.insert(t.clone());
                        beta_is_nullable = false;
                        break;
                    }
                    Symbol::NonTerminal(nt) => {
                        if let Some(first_set) = first_sets.get(nt) {
                            lookaheads.extend(first_set.iter().cloned());
                        }
                        if !nullable_set.contains(nt) {
                            beta_is_nullable = false;
                            break;
                        }
                    }
                }
            }

            // If β can be empty, add 'a' to the lookaheads.
            if beta_is_nullable {
                lookaheads.insert(a.clone());
            }

            // For each production B -> γ, and for each terminal b in FIRST(βa),
            // add [B -> .γ, b] to the closure.
            for prod in productions.iter().filter(|p| p.lhs == *b) {
                for l in &lookaheads {
                    let new_item = Item {
                        production: prod.clone(),
                        dot_position: 0,
                        lookahead: l.clone(),
                    };
                    if closure.insert(new_item.clone()) {
                        worklist.push_back(new_item);
                    }
                }
            }
        }
    }
    closure
}

pub fn compute_goto(items: &BTreeSet<Item>) -> BTreeSet<Item> {
    let mut result = BTreeSet::new();
    for item in items {
        if item.dot_position < item.production.rhs.len() {
            result.insert(Item {
                production: item.production.clone(),
                dot_position: item.dot_position + 1,
                lookahead: item.lookahead.clone(),
            });
        }
    }
    result
}

pub fn split_on_dot(items: &BTreeSet<Item>) -> BTreeMap<Option<Symbol>, BTreeSet<Item>> {
    let mut result: BTreeMap<Option<Symbol>, BTreeSet<Item>> = BTreeMap::new();
    for item in items {
        result
            .entry(item.production.rhs.get(item.dot_position).cloned())
            .or_default()
            .insert(item.clone());
    }
    result
}

