use crate::glr::grammar::{Production, Symbol};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap; // Added for derive macro pattern


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Item {
    pub production: Production,
    pub dot_position: usize,
}

// Manual impl for Item (could be derived)
impl JSONConvertible for Item {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("production".to_string(), self.production.to_json());
        obj.insert("dot_position".to_string(), self.dot_position.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let production = obj.remove("production").ok_or_else(|| "Missing field production for Item".to_string())
                                    .and_then(Production::from_json)?;
                let dot_position = obj.remove("dot_position").ok_or_else(|| "Missing field dot_position for Item".to_string())
                                      .and_then(usize::from_json)?;
                Ok(Item { production, dot_position })
            }
            _ => Err("Expected JSONNode::Object for Item".to_string()),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{nt as nt_sym, prod, t as t_sym, NonTerminal, Terminal}; // Renamed to avoid conflict

    // Helper to create a NonTerminal symbol easily in tests
    fn nt(name: &str) -> NonTerminal {
        NonTerminal(name.to_string())
    }

    // Helper to create a Terminal symbol easily in tests
    fn t(name: &str) -> Terminal {
        Terminal(name.to_string())
    }

    #[test]
    fn test_closure_with_nullable_nonterminal() {
        // Grammar:
        // S ::= A 'x'
        // A ::= 'y'
        // A ::= ε  (nullable)
        let productions = vec![
            prod("S", vec![nt_sym("A"), t_sym("x")]),
            prod("A", vec![t_sym("y")]),
            prod("A", vec![]), // Epsilon production for A
        ];

        // Initial item: S ::= . A 'x'
        let initial_item = Item {
            production: productions[0].clone(), // S ::= A 'x'
            dot_position: 0,
        };
        let initial_set = BTreeSet::from([initial_item.clone()]);

        let closure_set = compute_closure(&initial_set, &productions);

        // Expected closure:
        // S ::= . A 'x'
        // A ::= . 'y'
        // A ::= .       (representing A ::= . ε)
        let expected_item_s_ax = initial_item; // S ::= . A 'x'
        let expected_item_a_y = Item {
            production: productions[1].clone(), // A ::= 'y'
            dot_position: 0,
        };
        let expected_item_a_eps = Item {
            production: productions[2].clone(), // A ::= ε
            dot_position: 0,
        };

        let mut expected_closure = BTreeSet::new();
        expected_closure.insert(expected_item_s_ax);
        expected_closure.insert(expected_item_a_y);
        expected_closure.insert(expected_item_a_eps);

        assert_eq!(closure_set.len(), 3, "Closure should contain 3 items");
        assert_eq!(closure_set, expected_closure, "Closure set did not match expected LR(0) closure for nullable non-terminal");

        // Explicitly check that "S ::= A . 'x'" is NOT in the closure,
        // as this is handled by GOTO, not by LR(0) closure.
        let item_s_a_dot_x = Item {
            production: productions[0].clone(), // S ::= A 'x'
            dot_position: 1, // Dot after A
        };
        assert!(!closure_set.contains(&item_s_a_dot_x), "LR(0) closure should not advance dot over nullable non-terminal directly");
    }
}
