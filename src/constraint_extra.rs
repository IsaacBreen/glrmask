use crate::constraint::{
    GrammarConstraint, PrecomputeNode, PrecomputedNodeContents, LLMTokenBV,
    PrecomputedFinalizer
};
use crate::tokenizer::{TokenizerStateID, LLMTokenID};
use crate::types::TerminalID as GrammarTokenID;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::{Arc, Mutex};

// Helper function to format LLMTokenBV into a comma-separated string of LLMTokenIDs
fn format_llm_bv(bv: &LLMTokenBV) -> String {
    bv.iter_ones().map(|id| LLMTokenID(id).to_string()).collect::<Vec<_>>().join(", ")
}

// Helper function to format BTreeSet<GrammarTokenID> into a comma-separated string
fn format_grammar_token_set(set: &BTreeSet<GrammarTokenID>) -> String {
    set.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ")
}

// Helper function to format BTreeSet<TokenizerStateID> into a comma-separated string
fn format_tokenizer_state_set(set: &BTreeSet<TokenizerStateID>) -> String {
    set.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ")
}

// Recursive function to print the Trie structure for a single node and its children
fn print_trie_node_recursive(
    node: &PrecomputeNode,
    node_ptr_id: *const PrecomputeNode, // Use pointer for identity tracking and cycle detection
    indent: usize,
    visited: &mut BTreeSet<*const PrecomputeNode>, // Track visited nodes by pointer
    output: &mut String,
) -> std::fmt::Result {
    let prefix = "  ".repeat(indent);

    // Check if this node (identified by its memory address) has already been visited
    // in the current traversal path from the root.
    if visited.contains(&node_ptr_id) {
        // If visited, print a marker and stop recursion to prevent infinite loops in case of cycles.
        writeln!(output, "{prefix}Node @ {:p} (already visited, see above)", node_ptr_id)?;
        return Ok(());
    }
    // Mark the current node as visited for this path.
    visited.insert(node_ptr_id);

    // Print the node's address (useful for identifying shared nodes)
    writeln!(output, "{prefix}Node @ {:p}:", node_ptr_id)?;

    // Print the finalizers stored in the node's value
    if !node.value.finalizers.is_empty() {
        writeln!(output, "{prefix}  Finalizers:")?;
        for (i, finalizer) in node.value.finalizers.iter().enumerate() {
            writeln!(output, "{prefix}    Finalizer {}:", i)?;
            writeln!(output, "{prefix}      Possible Final Grammar Tokens: {{{}}}", format_grammar_token_set(&finalizer.possible_final_grammar_tokens))?;
            writeln!(output, "{prefix}      Compatible LLM Tokens: {{{}}}", format_llm_bv(&finalizer.compatible_llm_tokens))?;
            writeln!(output, "{prefix}      Tokenizer State IDs: {{{}}}", format_tokenizer_state_set(&finalizer.tokenizer_state_ids))?;
        }
    } else {
         writeln!(output, "{prefix}  No Finalizers")?;
    }

    // Print the outgoing edges from this node
    if !node.edges.is_empty() {
        writeln!(output, "{prefix}  Edges:")?;
        // Sort edges by GrammarTokenID for consistent output order
        let mut sorted_edges: Vec<_> = node.edges.iter().collect();
        sorted_edges.sort_by_key(|(grammar_token_id, _)| *grammar_token_id);

        for (grammar_token_id, edge_list) in sorted_edges {
             writeln!(output, "{prefix}    Grammar Token {}:", grammar_token_id)?;
             // Each grammar token can lead to multiple child nodes via different LLM token sets
             for (i, (edge_llm_bv, child_node_mutex)) in edge_list.iter().enumerate() {
                 // Get the raw pointer of the child node for identification
                 let child_node_ptr = Arc::as_ptr(child_node_mutex);
                 writeln!(output, "{prefix}      Edge {}: LLM Tokens {{{}}} -> Node @ {:p}", i, format_llm_bv(edge_llm_bv), child_node_ptr)?;

                 // Lock the child's mutex to access its data. Handle potential poisoning.
                 match child_node_mutex.lock() {
                    Ok(child_node) => {
                         // Recursively call print function for the child node
                         print_trie_node_recursive(&child_node, child_node_ptr, indent + 3, visited, output)?;
                    }
                    Err(poisoned) => {
                        // Log an error if the mutex was poisoned (shouldn't normally happen here)
                        writeln!(output, "{prefix}      Error: Mutex poisoned for child node @ {:p}. Details: {}", child_node_ptr, poisoned)?;
                        // Continue visualization if possible, or return error
                        // return Err(std::fmt::Error);
                    }
                 }
             }
        }
    } else {
        writeln!(output, "{prefix}  No Edges")?;
    }

    // Remove the current node from the visited set *after* processing its children.
    // This allows paths through this node from different branches if the graph isn't a strict tree.
    // If we want to print shared subtrees only once globally, `visited` should not be cleared here.
    // Keeping it this way prints the structure as reachable from each root/path.
    visited.remove(&node_ptr_id);

    Ok(())
}


impl GrammarConstraint {
    /// Generates a string visualizing the precomputed Trie structure for debugging.
    ///
    /// The visualization shows the Trie associated with each initial TokenizerStateID.
    /// Each node shows its memory address (for identifying shared nodes),
    /// finalizers (conditions under which paths ending here are valid), and
    /// outgoing edges labeled with GrammarTokenIDs and associated LLMToken BVs,
    /// leading to child nodes.
    pub fn visualize_precomputed(&self) -> String {
        let mut output = String::new();
        writeln!(output, "Precomputed Trie Visualization:").unwrap();
        writeln!(output, "Max LLM Token ID: {}", self.max_llm_token_id).unwrap();

        // Sort the tokenizer states for consistent output order
        let mut sorted_states: Vec<_> = self.precomputed.iter().collect();
        sorted_states.sort_by_key(|(tokenizer_state_id, _)| *tokenizer_state_id);

        for (tokenizer_state_id, root_node) in sorted_states {
            writeln!(output, "\n=== Tokenizer State ID: {} ===", tokenizer_state_id).unwrap();

            // Initialize an empty set to track visited nodes for cycle detection within this specific Trie traversal.
            let mut visited = BTreeSet::new();
            // Get the raw pointer to the root node for identity tracking
            let root_node_ptr: *const PrecomputeNode = root_node as *const _;

            // Start the recursive printing process from the root node
            if let Err(e) = print_trie_node_recursive(root_node, root_node_ptr, 0, &mut visited, &mut output) {
                 // Log any formatting errors encountered during visualization
                 writeln!(output, "Error during visualization for state {}: {}", tokenizer_state_id, e).unwrap();
            }
        }

        output
    }
}
