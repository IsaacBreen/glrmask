use crate::constraint::{GrammarConstraint, Precomputed, PrecomputeNode, PrecomputedNodeContents, PrecomputedFinalizer};
use crate::datastructures::trie::{Trie, node_ptr};
use crate::tokenizer::{TokenizerStateID, LLMTokenID};
use crate::types::TerminalID as GrammarTokenID; // Corrected import path
use crate::LLMTokenBV;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use bitvec::prelude::BitVec;

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
fn print_finalizer(finalizer: &PrecomputedFinalizer, indent: &str) {
    println!("{}  - Finalizer:", indent);
    let final_grammar_tokens: Vec<String> = finalizer.possible_final_grammar_tokens.iter().map(|id| id.0.to_string()).collect();
    println!("{}    Possible Final Grammar Tokens: [{}]", indent, final_grammar_tokens.join(", "));
    println!("{}    Compatible LLM Tokens: {}", indent, format_bv_indices(&finalizer.compatible_llm_tokens));
    let tokenizer_states: Vec<String> = finalizer.tokenizer_state_ids.iter().map(|id| id.0.to_string()).collect();
    println!("{}    Tokenizer States: [{}]", indent, tokenizer_states.join(", "));
}

/// Helper function to recursively dump the structure of a PrecomputeNode Trie.
fn dump_precompute_trie_recursive(
    node_arc: &Arc<Mutex<PrecomputeNode>>,
    indent: String,
    visited: &mut HashSet<*const PrecomputeNode>,
) {
    let node_ptr_val = node_ptr(node_arc);
    if !visited.insert(node_ptr_val) {
        println!("{}-> Ref {:?} (already printed)", indent, node_ptr_val);
        return;
    }

    let node = node_arc.lock().expect("Mutex poisoned during dump");

    println!("{}-> Node {:?} (MaxDepth: {})", indent, node_ptr_val, node.max_depth);

    // Print Node Value (Finalizers)
    if !node.value.finalizers.is_empty() {
        println!("{}  Value (Finalizers):", indent);
        for finalizer in &node.value.finalizers {
            print_finalizer(finalizer, &indent);
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
        for (edge_key, children_vec) in node.children() {
            for (edge_val_bv, child_arc) in children_vec {
                println!(
                    "{}Edge GrammarTokenID({}): LLM Tokens: {} -> Child Ptr: {:?}",
                    indent,
                    edge_key.0,
                    format_bv_indices(edge_val_bv),
                    node_ptr(child_arc)
                );
                // Recurse
                dump_precompute_trie_recursive(child_arc, new_indent.clone(), visited);
            }
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

            // Need to wrap the root_node_trie (which is a Trie, not an Arc<Mutex<Trie>>)
            // in an Arc<Mutex<>> to match the recursive function's expectation.
            // This is slightly awkward but necessary for the shared recursive logic.
            let root_node_arc = Arc::new(Mutex::new(root_node_trie.clone()));

            let mut visited: HashSet<*const PrecomputeNode> = HashSet::new();
            dump_precompute_trie_recursive(&root_node_arc, "".to_string(), &mut visited);
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }
}
