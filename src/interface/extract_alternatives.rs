use std::collections::{HashMap, HashSet};
use crate::glr::grammar::{Production, Symbol, NonTerminal};
use crate::interface::GrammarDefinition;

/// Extract complex alternatives (sequences with 2+ symbols) into separate helper rules.
/// This reduces the number of terminals in the final optimized grammar by enabling
/// better sharing of common patterns.
///
/// Example:
///   _mem69 ::= '"all"' ':' __opt____def68_safe__ | __opt___key_subgraphs__ ':' _json_object
/// becomes:
///   _mem69 ::= _mem69_alt0 | _mem69_alt1
///   _mem69_alt0 ::= '"all"' ':' __opt____def68_safe__
///   _mem69_alt1 ::= __opt___key_subgraphs__ ':' _json_object
pub fn extract_complex_alternatives(grammar: &mut GrammarDefinition) {
    // Group productions by LHS
    let mut prods_by_lhs: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, prod) in grammar.productions.iter().enumerate() {
        prods_by_lhs.entry(prod.lhs.0.clone())
            .or_insert_with(Vec::new)
            .push(idx);
    }
    
    // Find non-terminals with MIXED complexity alternatives
    // (some simple 1-symbol, some complex 2+ symbols)
    let mut to_extract: Vec<(String, Vec<usize>, Vec<usize>)> = Vec::new();
    for (nt, prod_indices) in &prods_by_lhs {
        if prod_indices.len() < 2 {
            continue; // Need at least 2 alternatives
        }
        
        // Separate simple (1 symbol) vs complex (2+ symbols) alternatives
        let mut simple_indices = Vec::new();
        let mut complex_indices = Vec::new();
        
        for &idx in prod_indices {
            let len = grammar.productions[idx].rhs.len();
            if len == 0 || len == 1 {
                simple_indices.push(idx);
            } else {
                complex_indices.push(idx);
            }
        }
        
        // Only extract if we have BOTH simple and complex alternatives
        if !simple_indices.is_empty() && !complex_indices.is_empty() {
            to_extract.push((nt.clone(), simple_indices, complex_indices));
        }
    }
    
    if to_extract.is_empty() {
        return;
    }
    
    crate::debug!(4, "Extracting complex alternatives from {} rules", to_extract.len());
    
    // Create helper rules and replace originals
    let mut new_productions: Vec<Production> = Vec::new();
    let mut extracted_productions: Vec<Production> = Vec::new();
    let mut helper_counter = 0;
    
    // Mark productions that will be extracted (complex ones only)
    let mut extracted_indices: HashSet<usize> = HashSet::new();
    for (_, _, complex_indices) in &to_extract {
        for &idx in complex_indices {
            extracted_indices.insert(idx);
        }
    }
    
    // Mark simple alternatives that will be kept inline
    let mut kept_inline_indices: HashSet<usize> = HashSet::new();
    for (_, simple_indices, _) in &to_extract {
        for &idx in simple_indices {
            kept_inline_indices.insert(idx);
        }
    }
    
    // First, copy productions that are neither extracted nor kept inline
    for (idx, prod) in grammar.productions.iter().enumerate() {
        if !extracted_indices.contains(&idx) && !kept_inline_indices.contains(&idx) {
            new_productions.push(prod.clone());
        }
    }
    
    // Now handle extracted rules
    for (nt, simple_indices, complex_indices) in &to_extract {
        // Add simple alternatives inline (keep original productions)
        for &idx in simple_indices {
            new_productions.push(grammar.productions[idx].clone());
        }
        
        // Extract complex alternatives into helper rules
        for &idx in complex_indices {
            let prod = &grammar.productions[idx];
            
            let helper_name = format!("{}_alt{}", nt, helper_counter);
            helper_counter += 1;
            
            // Create helper production
            extracted_productions.push(Production {
                lhs: NonTerminal(helper_name.clone()),
                rhs: prod.rhs.clone(),
            });
            
            // Add reference to helper in main rule
            new_productions.push(Production {
                lhs: NonTerminal(nt.clone()),
                rhs: vec![Symbol::NonTerminal(NonTerminal(helper_name))],
            });
        }
    }
    
    // Add all extracted helper productions
    new_productions.extend(extracted_productions);
    
    grammar.productions = new_productions;
    
    crate::debug!(4, "Extracted {} helper rules", helper_counter);
}
