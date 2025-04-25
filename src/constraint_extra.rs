use crate::constraint::{Precomputed, PrecomputedNodeContents, PrecomputedFinalizer, LLMTokenBV};
use crate::types::{TokenizerStateID, TerminalID as GrammarTokenID};
use crate::datastructures::trie::Trie;
use bitvec::prelude::BitVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

// Define the full type for clarity, as PrecomputeNode alias might be internal
type NodeType = Trie<GrammarTokenID, LLMTokenBV, PrecomputedNodeContents>;

/// Helper to get the raw pointer from an Arc<Mutex<Trie>>. Panics if the mutex is poisoned.
fn node_ptr(node_arc: &Arc<Mutex<NodeType>>) -> *const NodeType {
    let guard = node_arc.try_lock().expect("Mutex poisoned when getting node pointer");
    &*guard as *const _
}

/// Helper function to format a BitVec into a comma-separated string of set indices.
fn format_bitvec(bv: &LLMTokenBV) -> String {
    bv.iter_ones().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
}

/// Prints a human-readable representation of the Precomputed data structure.
/// Traverses the Trie for each TokenizerStateID using BFS and prints node info,
/// finalizers, and edges. Handles cycles.
pub fn print_precomputed(precomputed: &Precomputed) {
    println!("Precomputed Data Structure Visualization:");
    println!("=========================================");

    if precomputed.is_empty() {
        println!("(Precomputed map is empty)");
        println!("=========================================");
        return;
    }

    for (tokenizer_state_id, root_node_inner) in precomputed {
        println!("\n--- Tokenizer State ID Root: {:?} ---", tokenizer_state_id);

        // Use BFS for printing the Trie structure for this tokenizer state
        let mut queue: VecDeque<(Arc<Mutex<NodeType>>, usize)> = VecDeque::new(); // (Node Arc, Indent Level)
        let mut visited: HashSet<*const NodeType> = HashSet::new(); // Visited node pointers
        let mut node_ids: HashMap<*const NodeType, usize> = HashMap::new(); // Pointer -> Display ID
        let mut next_node_id = 0;

        // Wrap the root node value in Arc/Mutex for consistent handling in BFS
        let root_arc = Arc::new(Mutex::new(root_node_inner.clone()));
        let root_ptr = node_ptr(&root_arc);

        if visited.insert(root_ptr) {
            node_ids.insert(root_ptr, next_node_id);
            next_node_id += 1;
            queue.push_back((root_arc, 0));
        }

        while let Some((node_arc, indent_level)) = queue.pop_front() {
            let node_ptr = node_ptr(&node_arc);
            let current_node_id = *node_ids.get(&node_ptr).unwrap();
            let indent = "  ".repeat(indent_level);

            let node = node_arc.lock().expect("Mutex poisoned during print");

            println!("{}Node ID: {} (Ptr: {:?})", indent, current_node_id, node_ptr);
            println!("{}  Max Depth: {}", indent, node.max_depth);

            // Print Finalizers stored in the node's value
            if !node.value.finalizers.is_empty() {
                println!("{}  Finalizers:", indent);
                for (i, finalizer) in node.value.finalizers.iter().enumerate() {
                    let final_tokens_str = finalizer.possible_final_grammar_tokens.iter()
                        .map(|tid| format!("{:?}", tid))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let llm_tokens_str = format_bitvec(&finalizer.compatible_llm_tokens);
                    let tokenizer_states_str = finalizer.tokenizer_state_ids.iter()
                        .map(|tsid| format!("{:?}", tsid))
                        .collect::<Vec<_>>()
                        .join(", ");

                    println!("{}    [{}] Possible Grammar Tokens: {{{}}}", indent, i, final_tokens_str);
                    println!("{}        Compatible LLM Tokens: [{}]", indent, llm_tokens_str);
                    println!("{}        Resulting Tokenizer States: {{{}}}", indent, tokenizer_states_str);
                }
            }

            // Print Children (Edges)
            if !node.children().is_empty() {
                println!("{}  Edges:", indent);
                // Sort edges by GrammarTokenID (edge key) for consistent output
                let mut sorted_children: Vec<_> = node.children().iter().collect();
                sorted_children.sort_by_key(|(k, _)| *k);

                for (grammar_token_id, children_vec) in sorted_children {
                    for (edge_llm_tokens, child_arc) in children_vec { // child_arc is Arc<Mutex<NodeType>>
                        let child_ptr = node_ptr(child_arc);
                        let child_node_id = *node_ids.entry(child_ptr).or_insert_with(|| {
                            let id = next_node_id;
                            next_node_id += 1;
                            id
                        });

                        let edge_llm_str = format_bitvec(edge_llm_tokens);
                        println!("{}    - Via Grammar Token: {:?}, Edge LLM Tokens: [{}], -> Leads to Node ID: {} (Ptr: {:?})",
                                 indent, grammar_token_id, edge_llm_str, child_node_id, child_ptr);

                        if visited.insert(child_ptr) { // Add child to queue only if not visited
                            queue.push_back((child_arc.clone(), indent_level + 1));
                        }
                    }
                }
            } else {
                 println!("{}  No Edges (Leaf Node)", indent);
            }
            println!("{}---", indent); // Separator for nodes at the same level
        }
    }
    println!("=========================================");
}
