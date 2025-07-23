use crate::glr::grammar::{compute_epsilon_nonterminals, compute_first_sets, compute_follow_sets, NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap;
use std::fmt::{Display, Formatter};
// Added for derive macro pattern


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Item {
    pub production: Production,
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
                let production = obj.remove("production").ok_or_else(|| "Missing field production for Item".to_string())
                                    .and_then(Production::from_json)?;
                let dot_position = obj.remove("dot_position").ok_or_else(|| "Missing field dot_position for Item".to_string())
                                      .and_then(usize::from_json)?;
                let lookahead = obj.remove("lookahead").ok_or_else(|| "Missing field lookahead for Item".to_string())
                                     .and_then(Option::<Terminal>::from_json)?;
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
        write!(f, ", ")?;
        match &self.lookahead {
            Some(t) => write!(f, "{}", t)?,
            None => write!(f, "$")?,
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl Item {
    pub fn dot_at_end(&self) -> bool {
        self.dot_position == self.production.rhs.len()
    }

    pub fn next(&self) -> Self {
        // Create a new item with the dot moved one position to the right
        if self.dot_position < self.production.rhs.len() {
            Item {
                production: self.production.clone(),
                dot_position: self.dot_position + 1,
                lookahead: self.lookahead.clone(),
            }
        } else {
            // If the dot is at the end, return self
            self.clone()
        }
    }
}

pub fn compute_lookaheads(
    item: &Item,
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeSet<Terminal> {
    match item.production.rhs.get(item.dot_position + 1) {
        Some(Symbol::Terminal(t)) => {
            // If the next symbol is a terminal, the lookahead is just that terminal
            BTreeSet::from([t.clone()])
        }
        Some(Symbol::NonTerminal(nt)) => {
            let first_set = first_sets.get(nt).cloned().unwrap_or_default();
            if nullable_nonterminals.contains(nt) {
                // If the non-terminal is nullable, we also need to include the lookaheads for the next item
                let next_lookaheads = compute_lookaheads(
                    &item.next(),
                    productions,
                    first_sets,
                    nullable_nonterminals,
                );
                first_set.union(&next_lookaheads).cloned().collect()
            } else {
                first_set
            }
        }
        None => {
            // The child production is of length 0. Inherit the lookahead from the parent production.
            if let Some(lookahead) = &item.lookahead {
                BTreeSet::from([lookahead.clone()])
            } else {
                BTreeSet::new() // No lookahead if none is provided
            }
        }
    }
}

pub fn compute_closure(items: &BTreeSet<Item>, productions: &[Production]) -> BTreeSet<Item> {
    // crate::debug!(3, "Computing closure");
    let first_sets = compute_first_sets(productions);
    let nullable_nonterminals = compute_epsilon_nonterminals(productions);
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        if let Some(Symbol::NonTerminal(nt)) = item.production.rhs.get(item.dot_position) {
            for prod in productions.iter().filter(|p| p.lhs == *nt) {
                let lookaheads = compute_lookaheads(&item.next(), productions, &first_sets, &nullable_nonterminals);
                for lookahead in lookaheads {
                    let new_item = Item {
                        production: prod.clone(),
                        dot_position: 0,
                        lookahead: Some(lookahead),
                    };
                    if closure.insert(new_item.clone()) {
                        worklist.push_back(new_item);
                    }
                }
            }
        }
    }

    // crate::debug!(3, "Done computing closure");
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

