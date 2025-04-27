use crate::constraint::{GrammarConstraint, Precomputed, PrecomputeNode, PrecomputedNodeContents, PrecomputedFinalizer};
use crate::datastructures::trie::{Trie, node_ptr};
use crate::tokenizer::{TokenizerStateID, LLMTokenID};
use crate::types::TerminalID as GrammarTokenID;
use crate::constraint::LLMTokenBV;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use bitvec::prelude::BitVec; // Keep this import

/// Helper function to print the indices of set bits in a BitVec.
fn format_bv_indices(bv: &LLMTokenBV) -> String {
    let indices: Vec<String> = bv.iter_ones().map(|i| i.to_string()).collect();
    if indices.len() > 10 {
        format!("[{} indices starting with {}...]", indices.len(), indices[0..5].join(", "))
    } else if indices.is_empty() {
        "[]".to_string()
    }
     else {
        format!("[{}]", indices.join(", "))
    }
}

/// Helper function to print PrecomputedFinalizer details.
pub(crate) fn print_finalizer(grammar_token_id: GrammarTokenID, finalizer: &PrecomputedFinalizer, indent: &str) {
    println!("{}  - Finalizer for GrammarTokenID({}):", indent, grammar_token_id.0);
    println!("{}    Compatible LLM Tokens: {}", indent, format_bv_indices(finalizer.compatible_llm_tokens()));
    let tokenizer_states: Vec<String> = finalizer.tokenizer_state_ids().iter().map(|id| id.0.to_string()).collect();
    println!("{}    Tokenizer States: [{}]", indent, tokenizer_states.join(", "));
}

/// Helper function to recursively dump the structure of a PrecomputeNode Trie.
fn dump_precompute_trie_recursive(
    node_arc: &Arc<Mutex<PrecomputeNode>>,
    indent: String,
    visited: &mut HashSet<*const PrecomputeNode>, // Use the type alias directly
) {
    // Use the helper function from the trie module to get the pointer
    let node_ptr_val = match node_arc.try_lock() {
        Ok(guard) => &*guard as *const PrecomputeNode,
        Err(_) => {
            println!("{}-> Mutex poisoned for node, cannot get pointer.", indent);
            return; // Skip poisoned nodes
        }
    };

    if !visited.insert(node_ptr_val) {
        println!("{}-> Ref {:?} (already printed)", indent, node_ptr_val);
        return;
    }

    // Lock again to access content (handle potential poisoning)
    let node_lock_result = node_arc.lock();
    if node_lock_result.is_err() {
         println!("{}-> Mutex poisoned for node {:?}, cannot print content.", indent, node_ptr_val);
         return;
    }
    let node = node_lock_result.unwrap();


    println!("{}-> Node {:?} (MaxDepth: {})", indent, node_ptr_val, node.max_depth);

    // Print Node Value (Finalizers)
    if !node.value.finalizers().is_empty() {
        println!("{}  Value:", indent);
        for (grammar_token_id, finalizer) in node.value.finalizers() {
            print_finalizer(*grammar_token_id, finalizer, &indent);
        }
    } else {
         println!("{}  Value: (No finalizers)", indent);
    }

    // Print Children (Edges)
    if node.children().is_empty() {
        println!("{}  (Leaf Node)", indent);
    } else {
        println!("{}  Children:", indent);
        let new_indent = format!("{}    ", indent);
        // Collect children arcs first to avoid holding lock during recursion
        let children_to_visit: Vec<(GrammarTokenID, LLMTokenBV, Arc<Mutex<PrecomputeNode>>)> = node.children()
            .iter()
            .flat_map(|(edge_key, children_vec)| {
                children_vec.iter().map(move |(edge_val_bv, child_arc)| {
                    (*edge_key, edge_val_bv.clone(), child_arc.clone())
                })
            })
            .collect();

        // Drop the lock before recursing
        drop(node);

        for (edge_key, edge_val_bv, child_arc) in children_to_visit {
             // Get child pointer again safely
             let child_ptr_val = match child_arc.try_lock() {
                 Ok(guard) => &*guard as *const PrecomputeNode,
                 Err(_) => {
                     println!("{}Edge GrammarTokenID({}): LLM Tokens: {} -> Child Mutex Poisoned", indent, edge_key.0, format_bv_indices(&edge_val_bv));
                     continue; // Skip poisoned child
                 }
             };

            println!(
                "{}Edge GrammarTokenID({}): LLM Tokens: {} -> Child Ptr: {:?}",
                indent,
                edge_key.0,
                format_bv_indices(&edge_val_bv),
                child_ptr_val // Use the safely obtained pointer
            );
            // Recurse
            dump_precompute_trie_recursive(&child_arc, new_indent.clone(), visited);
        }
    }
}


impl GrammarConstraint {
    /// Dumps the structure of the precomputed Trie map for visualization.
    pub fn dump_precomputed(&self) {
        println!("Dumping Precomputed Trie Structure:");
        println!("===================================");

        for (tokenizer_state_id, root_node_trie) in &self.precomputed {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            // Wrap the root_node_trie in Arc<Mutex<>> for the recursive function.
            let root_node_arc = Arc::new(Mutex::new(root_node_trie.clone()));

            let mut visited: HashSet<*const PrecomputeNode> = HashSet::new();
            dump_precompute_trie_recursive(&root_node_arc, "".to_string(), &mut visited);
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }
}

#[cfg(test)]
mod tests {
    use crate::finite_automata::{eat_u8, Regex};
    use crate::glr::grammar::{prod, t, Terminal};
    use crate::glr::parser::GLRParser;
    use crate::glr::table::generate_glr_parser_with_terminal_map;
    use crate::tokenizer::{LLMTokenID, LLMTokenMap};
    use crate::types::TerminalID;
    use bimap::BiBTreeMap;
    use super::*;
    use bitvec::prelude::*;
    use crate::{choice, groups, seq}; // Added imports

    #[test]
    fn test_format_bv_indices_empty() {
        let bv = bitvec![usize, Lsb0;];
        assert_eq!(format_bv_indices(&bv), "[]");

        let bv = bitvec![usize, Lsb0; 0, 0, 0];
        assert_eq!(format_bv_indices(&bv), "[]");
    }

    #[test]
    fn test_format_bv_indices_single() {
        let mut bv = bitvec![usize, Lsb0; 0; 5];
        bv.set(3, true);
        assert_eq!(format_bv_indices(&bv), "[3]");
    }

    #[test]
    fn test_format_bv_indices_multiple_few() {
        let mut bv = bitvec![usize, Lsb0; 0; 10];
        bv.set(1, true);
        bv.set(5, true);
        bv.set(8, true);
        assert_eq!(format_bv_indices(&bv), "[1, 5, 8]");
    }

    #[test]
    fn test_format_bv_indices_multiple_many() {
        let mut bv = bitvec![usize, Lsb0; 0; 20];
        for i in 0..15 { bv.set(i, true); }
        assert_eq!(format_bv_indices(&bv), "[15 indices starting with 0, 1, 2, 3, 4...]");
    }

    // Helper function to create a minimal constraint for testing dump
    fn create_minimal_constraint() -> GrammarConstraint {
        // Tokenizer: Matches "a" (token 0), "aa" (token 1), "$" (token 2)
        let expr = groups![
            eat_u8(b'a'), // Grammar Token 0 ("A")
            seq![eat_u8(b'a'), eat_u8(b'a')], // Grammar Token 1 ("AA")
            eat_u8(b'$')  // Grammar Token 2 ("EOF")
        ];
        let tokenizer = expr.build();

        // LLM Token Map: "aa" -> 0, "$" -> 1
        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"aa".to_vec(), LLMTokenID(0)); // LLM Token 0
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));  // LLM Token 1
        let max_llm_token_id = 1;

        // Grammar: S -> AA $
        let productions = vec![
            prod("S", vec![t("AA"), t("EOF")]),
        ];

        // Map grammar terminals to the tokenizer's token IDs
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0)); // "a" from tokenizer - unused in grammar
        grammar_token_map.insert(Terminal("AA".to_string()), TerminalID(1)); // "aa" from tokenizer
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(2)); // "$" from tokenizer

        // Generate parser
        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);

        // Create constraint (this runs precomputation)
        GrammarConstraint::new(tokenizer, parser, llm_token_map, max_llm_token_id)
    }

    #[test]
    fn test_dump_precomputed_runs() {
        let constraint = create_minimal_constraint();
        println!("--- Starting dump_precomputed test output ---");
        // This test just ensures the dump function executes without panicking.
        // Manual inspection of the output is needed for verification.
        constraint.dump_precomputed();
        println!("--- Finished dump_precomputed test output ---");
    }
}
