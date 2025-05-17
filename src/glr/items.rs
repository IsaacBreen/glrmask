use crate::glr::grammar::{Production, Symbol};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Item {
    pub production: Production,
    pub dot_position: usize,
}

use crate::json_serialization::{JSONNode, JSONConvertible}; // Add this line

impl JSONConvertible for Item {
    fn to_json(&self) -> JSONNode {
        crate::json_serialization::struct_to_json_object(vec![
            ("production", self.production.to_json()),
            ("dot_position", self.dot_position.to_json()),
        ])
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        let map = crate::json_serialization::json_object_to_btreemap(node)?;
        Ok(Item {
            production: map.get("production").ok_or_else(|| "Missing 'production' field for Item".to_string()).and_then(Production::from_json)?,
            dot_position: map.get("dot_position").ok_or_else(|| "Missing 'dot_position' field for Item".to_string()).and_then(usize::from_json)?,
        })
    }
}

pub fn compute_closure(items: &BTreeSet<Item>, productions: &[Production]) -> BTreeSet<Item> {
    // crate::debug!(3, "Computing closure");
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        if let Some(Symbol::NonTerminal(nt)) = item.production.rhs.get(item.dot_position) {
            for prod in productions.iter().filter(|p| p.lhs == *nt) {
                let new_item = Item {
                    production: prod.clone(),
                    dot_position: 0,
                };
                if closure.insert(new_item.clone()) {
                    worklist.push_back(new_item);
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
