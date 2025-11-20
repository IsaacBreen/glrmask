use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::interface::GrammarDefinition;
use crate::finite_automata::{Expr, QuantifierType};
use std::collections::{BTreeMap, HashMap};
use bimap::BiBTreeMap;
use std::sync::Arc;

/// Optimizes the grammar by merging adjacent terminals that appear only once.
pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    let mut changed = true;
    while changed {
        changed = false;
        // 1. Count occurrences of each terminal
        let mut terminal_counts = HashMap::new();
        for prod in &grammar.productions {
            for symbol in &prod.rhs {
                if let Symbol::Terminal(t) = symbol {
                    *terminal_counts.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }

        // 2. Identify mergeable pairs and merge them
        // We'll do this by iterating over productions and modifying them in place if possible,
        // or building a new list of productions.
        // Since we need to update the grammar's terminal definitions as well, we might need to be careful.

        let mut new_productions = Vec::new();
        let mut merged_any_in_pass = false;
        let productions = std::mem::take(&mut grammar.productions);

        for prod in &productions {
            let mut new_rhs = Vec::new();
            let mut i = 0;
            while i < prod.rhs.len() {
                let symbol = &prod.rhs[i];
                
                // Check if we can merge current symbol with the next one
                if i + 1 < prod.rhs.len() {
                    let next_symbol = &prod.rhs[i+1];
                    
                    if let (Symbol::Terminal(t1), Symbol::Terminal(t2)) = (symbol, next_symbol) {
                        if terminal_counts.get(&t1) == Some(&1) && terminal_counts.get(&t2) == Some(&1) {
                            // Merge t1 and t2
                            let new_terminal = merge_terminals(&t1, &t2, grammar);
                            new_rhs.push(Symbol::Terminal(new_terminal));
                            i += 2; // Skip next symbol
                            merged_any_in_pass = true;
                            changed = true;
                            continue;
                        }
                    }
                }
                
                new_rhs.push(symbol.clone());
                i += 1;
            }
            new_productions.push(Production {
                lhs: prod.lhs.clone(),
                rhs: new_rhs,
            });
        }
        
        grammar.productions = new_productions;
        
        if merged_any_in_pass {
            // We might want to clean up unused terminals from the grammar definitions here,
            // but it's not strictly necessary for correctness, just for cleanliness.
            // The loop will continue until no more merges are possible.
        }
    }
}

fn merge_terminals(t1: &Terminal, t2: &Terminal, grammar: &mut GrammarDefinition) -> Terminal {
    // 1. Get Exprs for t1 and t2
    let expr1 = get_expr_for_terminal(t1, grammar);
    let expr2 = get_expr_for_terminal(t2, grammar);
    
    // 2. Create new Expr
    let new_expr = match (expr1.clone(), expr2.clone()) {
        (Expr::U8Seq(mut v1), Expr::U8Seq(v2)) => {
            v1.extend(v2);
            Expr::U8Seq(v1)
        },
        (e1, e2) => {
             Expr::Seq(vec![e1, e2])
        }
    };
    
    // 3. Create new Terminal
    // If both are literals, we might make a new literal terminal.
    // Otherwise, it's a regex terminal.
    let new_terminal = match (t1, t2) {
        (Terminal::Literal(l1), Terminal::Literal(l2)) => {
            let mut new_bytes = l1.clone();
            new_bytes.extend(l2);
            Terminal::Literal(new_bytes)
        },
        _ => {
             // Generate a new name
             let name1 = match t1 { Terminal::RegexName(n) => n.clone(), Terminal::Literal(l) => format!("{:?}", l) }; // Simplified name generation
             let name2 = match t2 { Terminal::RegexName(n) => n.clone(), Terminal::Literal(l) => format!("{:?}", l) };
             let new_name = format!("{}_{}", name1, name2); 
             // Ensure uniqueness? The grammar usually handles unique names, but here we are synthesizing.
             // Let's use a simpler approach or ensure uniqueness if needed.
             // For now, let's just use a combined name.
             Terminal::RegexName(new_name)
        }
    };
    
    // 4. Register new terminal in grammar
    // We need a new group_id
    let new_group_id = grammar.group_id_to_expr.keys().max().cloned().unwrap_or(0) + 1;
    
    match &new_terminal {
        Terminal::Literal(bytes) => {
            grammar.literal_to_group_id.insert(bytes.clone(), new_group_id);
        },
        Terminal::RegexName(name) => {
            // Check if name exists, if so, append index
            let mut final_name = name.clone();
            let mut idx = 1;
            while grammar.regex_name_to_group_id.contains_left(&final_name) {
                final_name = format!("{}_{}", name, idx);
                idx += 1;
            }
            let new_terminal_fixed = Terminal::RegexName(final_name.clone());
             grammar.regex_name_to_group_id.insert(final_name, new_group_id);
             // Update return value if name changed
             if let Terminal::RegexName(_) = new_terminal {
                 // This is a bit messy because we return 'new_terminal' which might have the old name.
                 // Let's just return the fixed one.
                 grammar.group_id_to_expr.insert(new_group_id, new_expr);
                 return new_terminal_fixed;
             }
        }
    }
    
    grammar.group_id_to_expr.insert(new_group_id, new_expr);
    
    new_terminal
}

fn get_expr_for_terminal(t: &Terminal, grammar: &GrammarDefinition) -> Expr {
    let group_id = match t {
        Terminal::Literal(bytes) => grammar.literal_to_group_id.get_by_left(bytes),
        Terminal::RegexName(name) => grammar.regex_name_to_group_id.get_by_left(name),
    };
    
    if let Some(gid) = group_id {
        grammar.group_id_to_expr.get(gid).cloned().unwrap_or(Expr::Epsilon) // Should not happen
    } else {
        // Fallback or error? 
        Expr::Epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{Production, Symbol, Terminal, NonTerminal};
    use crate::finite_automata::Expr;
    use crate::datastructures::u8set::U8Set;

    fn create_dummy_grammar() -> GrammarDefinition {
        // S -> A B
        // A -> "a"
        // B -> "b"
        let mut grammar = GrammarDefinition {
            productions: vec![
                Production {
                    lhs: NonTerminal("S".to_string()),
                    rhs: vec![
                        Symbol::Terminal(Terminal::literal(b"a".to_vec())),
                        Symbol::Terminal(Terminal::literal(b"b".to_vec())),
                    ],
                }
            ],
            start_production_id: 0,
            literal_to_group_id: BiBTreeMap::new(),
            regex_name_to_group_id: BiBTreeMap::new(),
            group_id_to_expr: BTreeMap::new(),
            ignore_terminal_id: None,
            external_name_to_group_id: BiBTreeMap::new(),
        };

        // Register terminals
        grammar.literal_to_group_id.insert(b"a".to_vec(), 1);
        grammar.group_id_to_expr.insert(1, Expr::U8Seq(b"a".to_vec()));

        grammar.literal_to_group_id.insert(b"b".to_vec(), 2);
        grammar.group_id_to_expr.insert(2, Expr::U8Seq(b"b".to_vec()));

        grammar
    }

    #[test]
    fn test_optimize_grammar_merges_literals() {
        let mut grammar = create_dummy_grammar();
        optimize_grammar(&mut grammar);

        // Should now be S -> "ab"
        assert_eq!(grammar.productions.len(), 1);
        let prod = &grammar.productions[0];
        assert_eq!(prod.rhs.len(), 1);
        
        if let Symbol::Terminal(Terminal::Literal(bytes)) = &prod.rhs[0] {
            assert_eq!(*bytes, b"ab".to_vec());
        } else {
            panic!("Expected merged literal terminal, got {:?}", prod.rhs[0]);
        }
        
        // Check if new terminal is registered
        assert!(grammar.literal_to_group_id.contains_left(&b"ab".to_vec()));
    }

    #[test]
    fn test_optimize_grammar_merges_regex() {
        let mut grammar = create_dummy_grammar();
        // Add a regex terminal
        // S -> R1 R2
        // R1 -> [a-z] (appears once)
        // R2 -> [0-9] (appears once)
        
        let r1_name = "R1".to_string();
        let r2_name = "R2".to_string();
        
        grammar.productions.push(Production {
            lhs: NonTerminal("S2".to_string()),
            rhs: vec![
                Symbol::Terminal(Terminal::RegexName(r1_name.clone())),
                Symbol::Terminal(Terminal::RegexName(r2_name.clone())),
            ],
        });
        
        grammar.regex_name_to_group_id.insert(r1_name.clone(), 3);
        grammar.group_id_to_expr.insert(3, Expr::U8Class(U8Set::from_u8(b'a'))); // Simplified regex
        
        grammar.regex_name_to_group_id.insert(r2_name.clone(), 4);
        grammar.group_id_to_expr.insert(4, Expr::U8Class(U8Set::from_u8(b'0'))); // Simplified regex
        
        optimize_grammar(&mut grammar);
        
        // Find S2 production
        let prod = grammar.productions.iter().find(|p| p.lhs.0 == "S2").expect("S2 production not found");
        assert_eq!(prod.rhs.len(), 1);
        
        if let Symbol::Terminal(Terminal::RegexName(name)) = &prod.rhs[0] {
            assert!(name.contains("R1"));
            assert!(name.contains("R2"));
            assert!(grammar.regex_name_to_group_id.contains_left(name));
        } else {
            panic!("Expected merged regex terminal, got {:?}", prod.rhs[0]);
        }
    }
}
