use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};

pub fn validate(productions: &[Production]) -> Result<(), String> {
    let mut lhs_nonterms: BTreeSet<NonTerminal> = BTreeSet::new();
    let mut rhs_nonterms: BTreeSet<NonTerminal> = BTreeSet::new();

    for prod in productions {
        lhs_nonterms.insert(prod.lhs.clone());
        for symbol in &prod.rhs {
            if let Symbol::NonTerminal(nt) = symbol {
                rhs_nonterms.insert(nt.clone());
            }
        }
    }

    let missing_nonterms: BTreeSet<_> = rhs_nonterms.difference(&lhs_nonterms).collect();
    if !missing_nonterms.is_empty() {
        let missing_nonterm_strings: Vec<_> = missing_nonterms.into_iter().map(|nt| nt.0.clone()).collect();
        return Err(format!("Nonterminals missing a production: {:?}.", missing_nonterm_strings));
    }

    Ok(())
}

pub fn drop_dead(productions: &[Production]) -> Vec<Production> {
    let start_prod = &productions[0];
    let mut reachable_nonterminals = BTreeSet::new();
    let mut worklist: Vec<_> = start_prod.rhs.iter().filter_map(|symbol| {
        if let Symbol::NonTerminal(nt) = symbol {
            Some(nt.clone())
        } else {
            None
        }
    }).collect();
    
    reachable_nonterminals.extend(worklist.clone());

    while let Some(nt) = worklist.pop() {
        for prod in productions {
            if prod.lhs == nt {
                for symbol in &prod.rhs {
                    if let Symbol::NonTerminal(next_nt) = symbol {
                        if reachable_nonterminals.insert(next_nt.clone()) {
                            worklist.push(next_nt.clone());
                        }
                    }
                }
            }
        }
    }

    let new_productions: Vec<_> = productions.iter()
        .filter(|prod| reachable_nonterminals.contains(&prod.lhs) || *prod == start_prod)
        .cloned()
        .collect();

    crate::debug!(2, "Dropped {} productions", productions.len() - new_productions.len());

    new_productions
}