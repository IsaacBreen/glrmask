//! Suffix Grammar Construction
//!
//! Transforms a grammar for language L into a grammar for Suf(L) = { y | ∃x: xy ∈ L }
//!
//! Algorithm:
//! 1. For each nonterminal A, create suffix nonterminal A• 
//! 2. For each terminal a, create helper T_a → a | ε
//! 3. For each production A → X₁...Xₖ, add: A• → sufSym(Xᵢ) X_{i+1}...Xₖ for i=1..k
//! 4. Start symbol is S• (suffix of original start symbol)

use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::GrammarDefinition;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use bimap::BiBTreeMap;
use crate::finite_automata::Expr;

/// Marker for suffix nonterminals
const SUFFIX_MARKER: &str = "_suffix";

/// Marker for terminal helper nonterminals
const TERMINAL_HELPER_PREFIX: &str = "_T_";

/// Create the suffix nonterminal name from an original nonterminal
fn suffix_nonterminal_name(name: &str) -> String {
    format!("{}{}", name, SUFFIX_MARKER)
}

/// Create the terminal helper nonterminal name
fn terminal_helper_name(terminal: &Terminal) -> String {
    match terminal {
        Terminal::Literal(bytes) => {
            // Use hex encoding for the literal bytes
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            format!("{}{}", TERMINAL_HELPER_PREFIX, hex)
        }
        Terminal::RegexName(name) => {
            format!("{}{}", TERMINAL_HELPER_PREFIX, name)
        }
    }
}

/// Get the suffix symbol for a grammar symbol
/// - For nonterminal B, returns B•
/// - For terminal a, returns T_a (helper nonterminal)
fn suffix_symbol(symbol: &Symbol) -> Symbol {
    match symbol {
        Symbol::NonTerminal(nt) => {
            Symbol::NonTerminal(NonTerminal(suffix_nonterminal_name(&nt.0)))
        }
        Symbol::Terminal(t) => {
            Symbol::NonTerminal(NonTerminal(terminal_helper_name(t)))
        }
    }
}

/// Transform a grammar for language L into a grammar for Suf(L)
///
/// Given a grammar G = (N, Σ, P, S), constructs G_suf = (N', Σ, P', S•) where:
/// - N' = N ∪ { A• | A ∈ N } ∪ { T_a | a ∈ Σ }
/// - P' includes:
///   - All original productions P (to generate the "tail" after the suffix start)
///   - Terminal helpers: T_a → a | ε for each terminal a
///   - Suffix rules: A• → sufSym(X_i) X_{i+1}...X_k for each A → X_1...X_k and 1 ≤ i ≤ k
///
/// # Example
/// For the grammar S → aSb | ε (generating a^n b^n):
/// - Adds T_a → a | ε, T_b → b | ε
/// - Adds S• → T_a S b | S• b | T_b (for S → aSb)
/// - Adds S• → ε (for S → ε)
///
/// The resulting grammar generates all suffixes of a^n b^n.
pub fn grammar_to_suffix_grammar(grammar: &GrammarDefinition) -> GrammarDefinition {
    let mut new_productions = grammar.productions.clone();
    let mut new_literal_to_group_id = grammar.literal_to_group_id.clone();
    let mut new_regex_name_to_group_id = grammar.regex_name_to_group_id.clone();
    let mut new_group_id_to_expr = grammar.group_id_to_expr.clone();
    
    // Track which terminals we've seen (to create helpers)
    let mut terminals_seen: BTreeSet<Terminal> = BTreeSet::new();
    
    // Collect all terminals from productions
    for prod in &grammar.productions {
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                terminals_seen.insert(t.clone());
            }
        }
    }
    
    // Find the next available group_id
    let mut next_group_id = grammar.group_id_to_expr.keys()
        .chain(grammar.literal_to_group_id.right_values())
        .chain(grammar.regex_name_to_group_id.right_values())
        .max()
        .copied()
        .unwrap_or(0) + 1;
    
    // Create terminal helper productions: T_a → a | ε
    // We model this as a nonterminal with productions for 'a' and 'ε'
    for terminal in &terminals_seen {
        let helper_name = terminal_helper_name(terminal);
        let helper_nt = NonTerminal(helper_name.clone());
        
        // T_a → a (just the terminal)
        new_productions.push(Production {
            lhs: helper_nt.clone(),
            rhs: vec![Symbol::Terminal(terminal.clone())],
        });
        
        // T_a → ε (empty string)
        new_productions.push(Production {
            lhs: helper_nt.clone(),
            rhs: vec![], // Empty RHS = epsilon
        });
        
        // Register the helper as a nonterminal (no group_id needed since it's a nonterminal)
    }
    
    // Create suffix productions
    for prod in &grammar.productions {
        let suffix_lhs = NonTerminal(suffix_nonterminal_name(&prod.lhs.0));
        
        if prod.rhs.is_empty() {
            // A → ε produces A• → ε
            new_productions.push(Production {
                lhs: suffix_lhs.clone(),
                rhs: vec![],
            });
        } else {
            // For A → X_1 X_2 ... X_k, add:
            // A• → sufSym(X_i) X_{i+1} ... X_k for i = 1..k
            for i in 0..prod.rhs.len() {
                let mut suffix_rhs = Vec::new();
                
                // The suffix symbol for position i (where suffix starts)
                suffix_rhs.push(suffix_symbol(&prod.rhs[i]));
                
                // Keep the rest of the symbols as-is (X_{i+1} ... X_k)
                for j in (i + 1)..prod.rhs.len() {
                    suffix_rhs.push(prod.rhs[j].clone());
                }
                
                new_productions.push(Production {
                    lhs: suffix_lhs.clone(),
                    rhs: suffix_rhs,
                });
            }
        }
    }
    
    // Find the new start production (S•) and move it to index 0
    // This is required because generate_glr_parser_with_maps hardcodes start_production_id = 0
    let original_start_name = &grammar.productions[grammar.start_production_id].lhs.0;
    let suffix_start_name = suffix_nonterminal_name(original_start_name);
    
    // Find the index of the first suffix start production
    let suffix_start_idx = new_productions
        .iter()
        .position(|p| p.lhs.0 == suffix_start_name)
        .expect("Suffix start production must exist");
    
    // Reorder productions: put suffix start first, then other suffix, then helpers, then original
    // This is required because generate_glr_parser_with_maps hardcodes start_production_id = 0
    let mut reordered_productions = Vec::new();
    
    // First, collect all suffix start productions (the ones for the start nonterminal's suffix)
    for prod in &new_productions {
        if prod.lhs.0 == suffix_start_name {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Then, collect all OTHER suffix productions (non-start suffix nonterminals)
    for prod in &new_productions {
        if prod.lhs.0.ends_with(SUFFIX_MARKER) && prod.lhs.0 != suffix_start_name {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Then, collect all terminal helper productions (needed by suffix productions)
    for prod in &new_productions {
        if prod.lhs.0.starts_with(TERMINAL_HELPER_PREFIX) {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Finally, collect original productions (needed for the "tail" after suffix)
    for prod in &new_productions {
        if !prod.lhs.0.ends_with(SUFFIX_MARKER) && !prod.lhs.0.starts_with(TERMINAL_HELPER_PREFIX) {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Now reordered_productions[0] should be a suffix start production
    let new_start_production_id = 0;
    
    GrammarDefinition {
        productions: reordered_productions,
        start_production_id: new_start_production_id,
        literal_to_group_id: new_literal_to_group_id,
        regex_name_to_group_id: new_regex_name_to_group_id,
        group_id_to_expr: new_group_id_to_expr,
        ignore_terminal_ids: grammar.ignore_terminal_ids.clone(),
        external_name_to_group_id: grammar.external_name_to_group_id.clone(),
    }
}

/// Validate terminal DWA paths against the suffix grammar.
/// 
/// Samples paths from the terminal DWA and checks what proportion are accepted
/// by the suffix parser. This validates that the terminal DWA isn't generating
/// spurious paths that don't correspond to valid grammar derivations.
///
/// # Arguments
/// * `dwa` - The terminal DWA to sample paths from
/// * `grammar` - The original grammar definition
/// * `terminals_count` - Number of terminals in the grammar (labels < this are terminal IDs)
/// * `num_samples` - Number of paths to sample
///
/// # Returns
/// The proportion of sampled paths that are accepted (0.0 to 1.0)
pub fn validate_terminal_dwa_paths(
    dwa: &crate::dwa_i32::DWA,
    grammar: &GrammarDefinition,
    terminals_count: usize,
    num_samples: usize,
) -> f64 {
    validate_terminal_dwa_paths_verbose(dwa, grammar, terminals_count, num_samples, false)
}

/// Verbose version of validate_terminal_dwa_paths that prints debug info
pub fn validate_terminal_dwa_paths_verbose(
    dwa: &crate::dwa_i32::DWA,
    grammar: &GrammarDefinition,
    terminals_count: usize,
    num_samples: usize,
    verbose: bool,
) -> f64 {
    use crate::interface::CompiledGrammar;
    use crate::glr::table::TerminalID;
    use rand::Rng;
    use std::sync::Arc;
    
    if verbose {
        println!("\n=== Original Grammar ===");
        println!("Productions:");
        for (i, prod) in grammar.productions.iter().enumerate() {
            let marker = if i == grammar.start_production_id { " <-- START" } else { "" };
            println!("  {}: {}{}", i, prod, marker);
        }
        println!("\nLiteral to group_id:");
        for (val, id) in &grammar.literal_to_group_id {
            println!("  {:?} -> {}", val, id);
        }
        println!("\nRegex name to group_id:");
        for (name, id) in &grammar.regex_name_to_group_id {
            println!("  {} -> {}", name, id);
        }
    }
    
    // Build suffix grammar and compile it
    let suffix_grammar = grammar_to_suffix_grammar(grammar);
    
    if verbose {
        println!("\n=== Suffix Grammar Productions ===");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            let marker = if i == suffix_grammar.start_production_id { " <-- START" } else { "" };
            println!("  {}: {}{}", i, prod, marker);
        }
    }
    
    let suffix_compiled = CompiledGrammar::from_definition(Arc::new(suffix_grammar));
    let suffix_parser = suffix_compiled.glr_parser();
    
    if verbose {
        println!("\n=== Suffix Parser Terminal Map ===");
        for (term, tid) in suffix_parser.terminal_map.iter() {
            println!("  {:?} -> TerminalID({})", term, tid.0);
        }
        
        // Print the parser table to understand what's happening
        println!("\n=== Suffix Parser Table (first few rows) ===");
        println!("{}", suffix_parser);
    }
    
    // Sample paths from the terminal DWA
    let mut rng = rand::thread_rng();
    let paths = dwa.sample_paths(num_samples, &mut rng);
    
    if verbose {
        println!("\n=== Terminal DWA Info ===");
        println!("  States: {}", dwa.states.len());
        println!("  Transitions: {}", dwa.states.num_transitions());
        println!("  Terminals count: {}", terminals_count);
        println!("  Sampled {} paths", paths.len());
    }
    
    if paths.is_empty() {
        return 1.0; // No paths = vacuously valid
    }
    
    let mut valid_count = 0;
    
    for (i, path) in paths.iter().enumerate() {
        // Extract terminal labels (filter out TSID labels which are >= terminals_count)
        let terminal_ids: Vec<TerminalID> = path
            .iter()
            .map(|(label, _state)| *label as usize)
            .filter(|&label| label < terminals_count)
            .map(|label| TerminalID(label))
            .collect();
        
        // Parse the terminal sequence with the suffix parser
        let state = suffix_parser.parse(&terminal_ids, None);
        let is_valid = state.is_ok();
        
        if verbose && i < 10 {
            let all_labels: Vec<_> = path.iter().map(|(l, _)| *l).collect();
            let term_labels: Vec<_> = terminal_ids.iter().map(|t| t.0).collect();
            println!("\nPath {}: all_labels={:?}, terminal_ids={:?}, valid={}", 
                     i, all_labels, term_labels, is_valid);
            
            // Debug: step through parsing terminal by terminal
            if !is_valid && !terminal_ids.is_empty() {
                let mut debug_state = suffix_parser.init_glr_parser(None);
                println!("  Initial: is_ok={}", debug_state.is_ok());
                for (j, tid) in terminal_ids.iter().enumerate() {
                    debug_state.step(*tid);
                    println!("  After step {}: terminal={}, is_ok={}", 
                             j, tid.0, debug_state.is_ok());
                }
            }
        }
        
        if is_valid {
            valid_count += 1;
        }
    }
    
    if verbose {
        println!("\nValid: {}/{} ({:.2}%)", valid_count, paths.len(), 
                 100.0 * valid_count as f64 / paths.len() as f64);
    }
    
    valid_count as f64 / paths.len() as f64
}

/// Get the original nonterminal name from a suffix nonterminal name
pub fn original_nonterminal_name(suffix_name: &str) -> Option<&str> {
    suffix_name.strip_suffix(SUFFIX_MARKER)
}

/// Check if a nonterminal name is a suffix nonterminal
pub fn is_suffix_nonterminal(name: &str) -> bool {
    name.ends_with(SUFFIX_MARKER)
}

/// Check if a nonterminal name is a terminal helper
pub fn is_terminal_helper(name: &str) -> bool {
    name.starts_with(TERMINAL_HELPER_PREFIX)
}

/// Prune a terminal DWA using the suffix grammar.
///
/// This walks the DWA and removes transitions that would lead to invalid
/// suffix parser states. A transition on terminal T is pruned if, after
/// feeding T to the suffix parser, the parser has no valid states.
///
/// The DWA labels are:
/// - 0..(terminals_count-1): terminal IDs  
/// - terminals_count..: TSID labels (tokenizer state IDs)
///
/// TSID transitions are not pruned (they represent tokenizer state changes,
/// not grammar terminals).
pub fn prune_dwa_with_suffix_grammar(
    dwa: &mut crate::dwa_i32::DWA,
    grammar: &GrammarDefinition,
    terminal_map: &bimap::BiBTreeMap<Terminal, crate::glr::table::TerminalID>,
    terminals_count: usize,
) -> (usize, usize) {
    use crate::interface::CompiledGrammar;
    use crate::glr::table::TerminalID;
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    use std::sync::Arc;
    
    crate::debug!(4, "Starting suffix grammar DWA pruning");
    
    // Build suffix grammar and compile it
    crate::debug!(4, "  Building suffix grammar...");
    let suffix_grammar = grammar_to_suffix_grammar(grammar);
    crate::debug!(4, "  Suffix grammar: {} productions", suffix_grammar.productions.len());
    
    crate::debug!(4, "  Compiling suffix grammar...");
    let suffix_compiled = CompiledGrammar::from_definition(Arc::new(suffix_grammar));
    crate::debug!(4, "  Suffix grammar compiled");
    
    let suffix_parser = suffix_compiled.glr_parser();
    
    // Build mapping from original terminal IDs to suffix parser terminal IDs
    // The suffix parser may have different terminal IDs
    let mut orig_to_suffix_tid: BTreeMap<usize, TerminalID> = BTreeMap::new();
    for (term, orig_tid) in terminal_map.iter() {
        // Look up the same terminal in the suffix parser
        if let Some(suffix_tid) = suffix_parser.terminal_map.get_by_left(term) {
            orig_to_suffix_tid.insert(orig_tid.0, *suffix_tid);
        }
    }
    
    // Track which suffix parser states are reachable at each DWA state
    // Key: DWA state ID, Value: Set of GLR parser state IDs
    type ParserStateSet = BTreeSet<crate::glr::table::StateID>;
    let mut dwa_to_parser_states: BTreeMap<usize, ParserStateSet> = BTreeMap::new();
    
    // Initialize: start DWA state maps to initial suffix parser state
    let initial_parser_state_id = suffix_parser.start_state_id;
    dwa_to_parser_states.insert(dwa.body.start_state, {
        let mut set = BTreeSet::new();
        set.insert(initial_parser_state_id);
        set
    });
    
    // BFS to propagate parser states through the DWA
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut visited: BTreeSet<usize> = BTreeSet::new();
    queue.push_back(dwa.body.start_state);
    visited.insert(dwa.body.start_state);
    
    // Track transitions to remove: (from_state, label)
    let mut transitions_to_remove: Vec<(usize, i32)> = Vec::new();
    let mut pruned_count = 0;
    let mut kept_count = 0;
    
    while let Some(dwa_state) = queue.pop_front() {
        let parser_states = dwa_to_parser_states.get(&dwa_state).cloned().unwrap_or_default();
        
        // For each transition from this DWA state
        let transitions: Vec<(i32, usize)> = dwa.states[dwa_state].transitions.iter()
            .map(|(&label, &dest)| (label, dest))
            .collect();
        
        for (label, dest_dwa_state) in transitions {
            let label_usize = label as usize;
            
            // Skip TSID transitions (not grammar terminals)
            if label_usize >= terminals_count {
                // TSID transition - always keep, parser state propagates unchanged
                let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                for &ps in &parser_states {
                    dest_states.insert(ps);
                }
                if !visited.contains(&dest_dwa_state) {
                    visited.insert(dest_dwa_state);
                    queue.push_back(dest_dwa_state);
                }
                kept_count += 1;
                continue;
            }
            
            // Terminal transition - check if suffix parser can accept it
            if let Some(&suffix_tid) = orig_to_suffix_tid.get(&label_usize) {
                // Check if ANY parser state can accept this terminal
                let mut any_valid = false;
                
                for &parser_state_id in &parser_states {
                    if let Some(row) = crate::glr::table::get_row(&suffix_parser.table, parser_state_id) {
                        // Check if there's any shift or reduce action for this terminal
                        let has_action = row.get_shifts_and_reduces_for_terminal(&suffix_tid).is_some();
                        if has_action {
                            any_valid = true;
                            break;
                        }
                    }
                }
                
                if any_valid {
                    // Keep the transition, propagate parser states
                    // (We're being conservative - keeping all states that could reach here)
                    let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                    for &ps in &parser_states {
                        dest_states.insert(ps);
                    }
                    if !visited.contains(&dest_dwa_state) {
                        visited.insert(dest_dwa_state);
                        queue.push_back(dest_dwa_state);
                    }
                    kept_count += 1;
                } else {
                    // Prune the transition
                    transitions_to_remove.push((dwa_state, label));
                    pruned_count += 1;
                }
            } else {
                // Terminal not found in suffix parser - this shouldn't happen
                // Keep the transition to be safe
                let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                for &ps in &parser_states {
                    dest_states.insert(ps);
                }
                if !visited.contains(&dest_dwa_state) {
                    visited.insert(dest_dwa_state);
                    queue.push_back(dest_dwa_state);
                }
                kept_count += 1;
            }
        }
    }
    
    // Remove pruned transitions
    for (from_state, label) in &transitions_to_remove {
        dwa.states[*from_state].transitions.remove(label);
        dwa.states[*from_state].trans_weights.remove(label);
    }
    
    crate::debug!(4, "Suffix grammar DWA pruning: kept={}, pruned={}", kept_count, pruned_count);
    
    (kept_count, pruned_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::GrammarDefinition;
    
    /// Test the suffix grammar construction on a recursive grammar
    /// Note: Currently skipped due to EBNF parser limitations with single-rule recursive grammars
    #[test]
    #[ignore]
    fn test_suffix_grammar_recursive() {
        // Parse a grammar with recursion: A → aAb | c
        let ebnf = r#"
            A ::= 'a' A 'b' | 'c';
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        
        // Convert to suffix grammar
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        // Print for debugging
        println!("Original productions:");
        for (i, prod) in grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        println!("\nSuffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Verify we have the expected structure
        let prod_strs: Vec<String> = suffix_grammar.productions.iter()
            .map(|p| format!("{}", p))
            .collect();
        
        // Should have terminal helpers
        assert!(prod_strs.iter().any(|s| s.contains("_T_") && s.contains("61")), // 'a' = 0x61
            "Should have terminal helper for 'a'");
        assert!(prod_strs.iter().any(|s| s.contains("_T_") && s.contains("62")), // 'b' = 0x62
            "Should have terminal helper for 'b'");
        
        // Should have suffix rules
        assert!(prod_strs.iter().any(|s| s.contains("A_suffix")),
            "Should have suffix nonterminal A_suffix");
    }
    
    /// Test suffix grammar on a simple grammar
    #[test]
    fn test_suffix_grammar_simple() {
        let ebnf = r#"
            start ::= "abc";
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Simple grammar suffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // The suffix grammar for "abc" should accept: "abc", "bc", "c", ""
        // This is modeled by creating helpers for the terminal "abc" and allowing
        // starting at any position within the expansion.
    }
    
    /// Test suffix grammar with alternation
    #[test]
    fn test_suffix_grammar_alternation() {
        let ebnf = r#"
            start ::= "a" | "b";
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Alternation grammar suffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Should have suffix rules for both alternatives
        let has_a_suffix = suffix_grammar.productions.iter()
            .any(|p| p.lhs.0.ends_with("_suffix"));
        assert!(has_a_suffix, "Should have suffix productions");
    }
    
    /// Test suffix grammar with sequence
    #[test]
    fn test_suffix_grammar_sequence() {
        let ebnf = r#"
            start ::= 'a' 'b' 'c';
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Sequence grammar suffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Should have suffix rules for each position in the sequence:
        // start_suffix -> T_a 'b' 'c'  (suffix starts in first element)
        // start_suffix -> T_b 'c'      (suffix starts in second element)
        // start_suffix -> T_c          (suffix starts in third element)
        let prod_strs: Vec<String> = suffix_grammar.productions.iter()
            .map(|p| format!("{}", p))
            .collect();
        
        // Count how many start_suffix productions we have
        let suffix_count = prod_strs.iter()
            .filter(|s| s.starts_with("start_suffix ->"))
            .count();
        
        // We should have 3 suffix productions for the 3-element sequence
        assert_eq!(suffix_count, 3, "Should have 3 suffix productions for 3-element sequence");
    }
    
    /// Test that suffix grammar preserves terminal helpers correctly
    #[test]
    fn test_suffix_grammar_terminal_helpers() {
        let ebnf = r#"
            start ::= 'hello';
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Terminal helper test productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Check that we have exactly 2 terminal helper productions:
        // T_hello -> 'hello'  (the terminal itself)
        // T_hello ->          (epsilon)
        let helper_prods: Vec<_> = suffix_grammar.productions.iter()
            .filter(|p| p.lhs.0.starts_with("_T_"))
            .collect();
        
        assert_eq!(helper_prods.len(), 2, "Should have exactly 2 terminal helper productions");
        
        // One should be non-empty (terminal) and one empty (epsilon)
        let non_empty = helper_prods.iter().filter(|p| !p.rhs.is_empty()).count();
        let empty = helper_prods.iter().filter(|p| p.rhs.is_empty()).count();
        assert_eq!(non_empty, 1, "Should have 1 non-empty terminal helper production");
        assert_eq!(empty, 1, "Should have 1 empty (epsilon) terminal helper production");
    }
}
