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
}

pub fn compute_first_set_for_item(
    item: &Item,
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeSet<Option<Terminal>> {
    if let Some((symbol, next_item)) = item.next() {
        match symbol {
            Symbol::Terminal(t) => {
                // If the next symbol is a terminal, the first is just that terminal
                BTreeSet::from([Some(t)])
            }
            Symbol::NonTerminal(nt) => {
                let mut first_set: BTreeSet<_> = first_sets.get(&nt).cloned().unwrap_or_default().into_iter()
                    .map(Some)
                    .collect();

                if nullable_nonterminals.contains(&nt) {
                    // If the non-terminal is nullable, we also need to include the firsts for the next item
                    let next_firsts = compute_first_set_for_item(
                        &next_item,
                        productions,
                        first_sets,
                        nullable_nonterminals,
                    );
                    first_set.extend(next_firsts);
                }
                first_set
            }
        }
    } else {
        // The dot is at the end. The first is the lookahead.
        BTreeSet::from([item.lookahead.clone()])
    }
}

pub enum LRType {
    LALR,
    LR1,
}

pub const LR_TYPE: LRType = LRType::LALR;

pub fn compute_closure(
    items: &BTreeSet<Item>,
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
    follow_sets: &BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>>,

) -> BTreeSet<Item> {
    // crate::debug!(3, "Computing closure");
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        if let Some((Symbol::NonTerminal(nt), next_item)) = item.next() {
            for prod in productions.iter().filter(|p| p.lhs == nt) {
                let lookaheads = compute_first_set_for_item(&next_item, productions, &first_sets, &nullable_nonterminals);
                for lookahead in lookaheads {
                    let new_item = Item {
                        production: prod.clone(),
                        dot_position: 0,
                        lookahead,
                    };
                    if closure.insert(new_item.clone()) {
                        worklist.push_back(new_item);
                    }
                }
            }
        }
    }

    if matches!(LR_TYPE, LRType::LALR) {
        let mut lalr_closure = BTreeSet::new();
        let mut reduce_item_cores: BTreeMap<(Production, usize), BTreeSet<Option<Terminal>>> = BTreeMap::new();

        // Separate reduce and non-reduce items, and group reduce items by core
        for item in closure {
            if item.dot_at_end() {
                reduce_item_cores.entry((item.production, item.dot_position)).or_default();
            } else {
                lalr_closure.insert(item);
            }
        }

        // Process reduce items by replacing their specific lookaheads with the full FOLLOW set.
        for ((prod, dot_pos), _) in reduce_item_cores {
            if let Some(follows) = follow_sets.get(&prod.lhs) {
                for lookahead in follows {
                    lalr_closure.insert(Item { production: prod.clone(), dot_position: dot_pos, lookahead: lookahead.clone() });
                }
            }
        }
        return lalr_closure;
    }
    closure
}

pub fn compute_goto(items: &BTreeSet<Item>) -> BTreeSet<Item> {
    items.iter()
        .filter_map(|item| item.next())
        .map(|(_, next_item)| next_item)
        .collect()
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
