use std::collections::BTreeMap; // Use BTreeMap to keep children sorted by edge label (byte vec)
use std::fmt;

// Represents a node in the VocabTrie
#[derive(PartialEq)] // Add PartialEq for easier testing
pub struct VocabTrieNode {
    /// The token ID if the path from the root to this node represents a complete token.
    /// The root node will have None unless the empty string is a token.
    token_id: Option<u32>,
    /// Children nodes, keyed by the byte vector representing the edge label.
    /// BTreeMap ensures edges are sorted lexicographically by byte vector,
    /// which is crucial for the merging algorithm.
    children: BTreeMap<Vec<u8>, VocabTrieNode>,
}

impl VocabTrieNode {
    /// Creates a new node, typically representing a token endpoint.
    fn new(token_id: Option<u32>) -> Self {
        VocabTrieNode {
            token_id,
            children: BTreeMap::new(),
        }
    }
}

// Manual implementation of Debug to handle potentially large byte vecs nicely
impl fmt::Debug for VocabTrieNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Custom formatting helper for byte vectors
        fn format_bytes(bytes: &[u8]) -> String {
            // Limit displayed bytes for readability
            const MAX_BYTES_DISPLAY: usize = 10;
            let display_str = String::from_utf8_lossy(bytes.get(..MAX_BYTES_DISPLAY).unwrap_or(bytes));
            if bytes.len() > MAX_BYTES_DISPLAY {
                format!("{}...({} bytes)", display_str, bytes.len())
            } else {
                format!("{}", display_str)
            }
        }

        let mut debug_struct = f.debug_struct("VocabTrieNode");
        debug_struct.field("token_id", &self.token_id);

        // Format children concisely
        let children_summary: BTreeMap<String, &VocabTrieNode> = self
            .children
            .iter()
            .map(|(k, v)| (format_bytes(k), v))
            .collect();
        debug_struct.field("children", &children_summary);

        debug_struct.finish()
    }
}


// The main VocabTrie structure
#[derive(Debug, PartialEq)]
pub struct VocabTrie {
    root: VocabTrieNode,
    // Optional: Store the conventional root token ID if needed,
    // but the root node itself usually represents the empty prefix.
    // root_token_id: Option<u32>,
}

impl VocabTrie {
    /// Creates an empty VocabTrie.
    pub fn new() -> Self {
        VocabTrie {
            // Root node represents the empty prefix, typically no token ID unless specified
            root: VocabTrieNode::new(None),
        }
    }

    /// Builds the VocabTrie from a list of tokens.
    /// Tokens are provided as (token_id, byte_vector) pairs.
    pub fn build(tokens: &[(u32, Vec<u8>)]) -> Self {
        let mut trie = VocabTrie::new();

        // 1. Initial population: Add all tokens as direct children of the root.
        //    Each edge is the full token byte vec, leading to a leaf node with the ID.
        for (id, bytes) in tokens {
            // Handle empty string token case - assign ID to root?
            // For now, assume tokens are non-empty or handle root ID separately if needed.
            if bytes.is_empty() {
                // Decide how to handle empty string token ID. Assign to root?
                // Current VocabTrieNode::new(None) for root assumes no empty token ID.
                // Let's assign it if we encounter one.
                 trie.root.token_id = Some(*id);
                continue;
            }
            // Insert node, potentially overwriting if duplicate byte vecs exist (last ID wins)
            trie.root
                .children
                .insert(bytes.clone(), VocabTrieNode::new(Some(*id)));
        }

        // 2. Merge nodes recursively starting from the root's children.
        Self::merge_nodes(&mut trie.root);

        trie
    }

    /// Recursively merges nodes according to the specified algorithm.
    /// Assumes `node.children` is sorted (guaranteed by BTreeMap).
    fn merge_nodes(node: &mut VocabTrieNode) {
        if node.children.len() <= 1 {
            // Base case: No merging needed if 0 or 1 child.
            // Still need to recurse down in case the single child needs merging internally.
            for child_node in node.children.values_mut() {
                Self::merge_nodes(child_node);
            }
            return;
        }

        // Take ownership of the children map to rebuild it.
        let old_children = std::mem::take(&mut node.children);
        let mut new_children = BTreeMap::new();

        // Use an iterator to process children in sorted order.
        let mut iter = old_children.into_iter().peekable();

        while let Some((mut marker_label, mut marker_node)) = iter.next() {
            // `marker_node` is the potential parent for subsequent nodes with prefixes.

            // Check subsequent nodes to see if they should be children of `marker_node`.
            while let Some((current_label, _)) = iter.peek() {
                // Check if current_label is prefixed by marker_label
                if current_label.starts_with(&marker_label) {
                    // Yes, this node should be a child of marker_node.
                    // Consume the current item from the iterator.
                    let (current_label_owned, current_node) = iter.next().unwrap();

                    // Calculate the suffix for the new edge label.
                    // Ensure slicing is safe (it should be due to starts_with).
                    let suffix = current_label_owned[marker_label.len()..].to_vec();

                    // Add the current_node as a child of marker_node using the suffix as edge label.
                    // Note: If the suffix is empty (duplicate token bytes), this might overwrite.
                    // The problem description doesn't specify duplicate handling; BTreeMap handles it.
                    if !suffix.is_empty() {
                         marker_node.children.insert(suffix, current_node);
                    } else {
                        // This case implies current_label == marker_label.
                        // This shouldn't happen if input token bytes are unique.
                        // If duplicates are allowed, decide how to merge (e.g., keep marker's ID).
                        // For now, we effectively discard the duplicate's node structure here,
                        // keeping only the marker_node.
                        // A different strategy might be needed depending on requirements.
                         eprintln!("Warning: Duplicate token bytes found: {:?}", marker_label);
                    }

                } else {
                    // No prefix match, this node starts a new group. Break the inner loop.
                    break;
                }
            }

            // After potentially adding children to marker_node, recursively merge *its* children.
            Self::merge_nodes(&mut marker_node);

            // Add the (potentially updated) marker_node to the new children map for the current level.
            new_children.insert(marker_label, marker_node);
        }

        // Replace the original node's children with the newly structured map.
        node.children = new_children;
    }

     /// Finds the token ID corresponding to the exact byte sequence.
    pub fn find_token(&self, bytes: &[u8]) -> Option<u32> {
        if bytes.is_empty() {
            // Handle lookup for the empty string token potentially stored at the root
            return self.root.token_id;
        }

        let mut current_node = &self.root;
        let mut remaining_bytes = bytes;

        loop {
            let mut found_match = false;
            // Iterate through children (sorted by BTreeMap, but order doesn't matter for lookup)
            for (edge_label, child_node) in &current_node.children {
                if remaining_bytes.starts_with(edge_label) {
                    // Found an edge matching a prefix of the remaining bytes
                    remaining_bytes = &remaining_bytes[edge_label.len()..];
                    current_node = child_node;
                    found_match = true;
                    break; // Move to the next level of the trie
                }
            }

            if !found_match {
                // No child edge matches the start of remaining_bytes
                return None;
            }

            if remaining_bytes.is_empty() {
                // We have consumed all bytes and landed on current_node.
                // Return its token_id (which might be None if it's an internal node
                // created implicitly, though the described algorithm avoids this).
                return current_node.token_id;
            }
        }
    }
}

impl Default for VocabTrie {
    fn default() -> Self {
        Self::new()
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create byte vecs from strings easily
    fn b(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn test_empty_trie() {
        let tokens: Vec<(u32, Vec<u8>)> = vec![];
        let trie = VocabTrie::build(&tokens);
        assert_eq!(trie.root.token_id, None);
        assert!(trie.root.children.is_empty());
        assert_eq!(trie.find_token(b"a"), None);
    }

    #[test]
    fn test_single_token() {
        let tokens = vec![(1, b("hello"))];
        let trie = VocabTrie::build(&tokens);

        assert_eq!(trie.root.token_id, None);
        assert_eq!(trie.root.children.len(), 1);
        assert!(trie.root.children.contains_key(&b("hello")));

        let node = trie.root.children.get(&b("hello")).unwrap();
        assert_eq!(node.token_id, Some(1));
        assert!(node.children.is_empty());

        assert_eq!(trie.find_token(b"hello"), Some(1));
        assert_eq!(trie.find_token(b"hell"), None);
        assert_eq!(trie.find_token(b"hello world"), None);
    }

     #[test]
    fn test_empty_string_token() {
        let tokens = vec![(0, b("")), (1, b("a"))];
        let trie = VocabTrie::build(&tokens);

        assert_eq!(trie.root.token_id, Some(0)); // Root gets the ID for ""
        assert_eq!(trie.root.children.len(), 1);
        assert!(trie.root.children.contains_key(&b("a")));
        assert_eq!(trie.root.children[&b("a")].token_id, Some(1));

        assert_eq!(trie.find_token(&b("")), Some(0));
        assert_eq!(trie.find_token(&b("a")), Some(1));
    }

    #[test]
    fn test_simple_prefix() {
        // "a" is prefix of "ab"
        let tokens = vec![(1, b("a")), (2, b("ab"))];
        let trie = VocabTrie::build(&tokens);

        // Expected structure: root --"a"--> Node(1) --"b"--> Node(2)
        assert_eq!(trie.root.children.len(), 1); // Only "a" edge from root
        assert!(trie.root.children.contains_key(&b("a")));

        let node_a = trie.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, Some(1)); // Node for "a" has ID 1
        assert_eq!(node_a.children.len(), 1); // Node "a" has one child
        assert!(node_a.children.contains_key(&b("b"))); // Edge is the suffix "b"

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, Some(2)); // Node for "ab" has ID 2
        assert!(node_ab.children.is_empty());

        assert_eq!(trie.find_token(&b("a")), Some(1));
        assert_eq!(trie.find_token(&b("ab")), Some(2));
        assert_eq!(trie.find_token(&b("b")), None);
        assert_eq!(trie.find_token(&b("abc")), None);
    }

    #[test]
    fn test_multiple_prefixes() {
        // "a", "ab", "abc"
        let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("abc"))];
        let trie = VocabTrie::build(&tokens);

        // Expected: root --"a"--> Node(1) --"b"--> Node(2) --"c"--> Node(3)
        assert_eq!(trie.root.children.len(), 1);
        let node_a = trie.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, Some(1));
        assert_eq!(node_a.children.len(), 1);

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, Some(2));
        assert_eq!(node_ab.children.len(), 1);

        let node_abc = node_ab.children.get(&b("c")).unwrap();
        assert_eq!(node_abc.token_id, Some(3));
        assert!(node_abc.children.is_empty());

        assert_eq!(trie.find_token(&b("a")), Some(1));
        assert_eq!(trie.find_token(&b("ab")), Some(2));
        assert_eq!(trie.find_token(&b("abc")), Some(3));
        assert_eq!(trie.find_token(&b("b")), None);
        assert_eq!(trie.find_token(&b("abcd")), None);
    }

    #[test]
    fn test_shared_prefix_branching() {
        // "apple", "apply" - share "appl" prefix
        let tokens = vec![(10, b("apple")), (20, b("apply"))];
        let trie = VocabTrie::build(&tokens);

        // Expected: root --"appl"--> Node(?) --"e"--> Node(10)
        //                      |
        //                      --"y"--> Node(20)
        // The intermediate node for "appl" should NOT exist / have an ID
        // because "appl" itself is not a token in the input.
        // Let's trace the algorithm:
        // 1. Root children: {"apple": Node(10), "apply": Node(20)} (sorted)
        // 2. merge_nodes(&root):
        //    - marker = ("apple", Node(10))
        //    - peek = ("apply", Node(20))
        //    - "apply".starts_with("apple") -> false
        //    - Add ("apple", Node(10)) to new_children. Recurse on Node(10) (no-op).
        //    - marker = ("apply", Node(20))
        //    - peek = None
        //    - Add ("apply", Node(20)) to new_children. Recurse on Node(20) (no-op).
        //    - root.children = {"apple": Node(10), "apply": Node(20)}
        // This seems wrong based on the radix tree idea. The user's algorithm description
        // needs careful interpretation. Let's re-read:
        // "If the vec at this index is prefixed by the vec at the marker, advance index and continue."
        // "Otherwise, we move the edges between the marker and the current index (excluding both)
        //  to the destination node of the edge at the marker. The marker edge's byte vec should be
        //  a prefix of all byte vecs we're moving. Remove that prefix when we move the edges."
        //
        // Let's re-simulate with the code's logic:
        // Input: [(10, b("apple")), (20, b("apply"))] (Assume sorted for BTreeMap)
        // Initial root.children: { b("apple"): Node(10), b("apply"): Node(20) }
        // merge_nodes(&root):
        //   iter.next() -> marker = (b("apple"), Node(10))
        //   iter.peek() -> Some((b("apply"), Node(20)))
        //   b("apply").starts_with(b("apple")) -> false. Break inner loop.
        //   merge_nodes(&mut marker_node) -> no-op
        //   new_children.insert(b("apple"), Node(10))
        //   iter.next() -> marker = (b("apply"), Node(20))
        //   iter.peek() -> None. End inner loop.
        //   merge_nodes(&mut marker_node) -> no-op
        //   new_children.insert(b("apply"), Node(20))
        //   root.children = new_children = { b("apple"): Node(10), b("apply"): Node(20) }
        //
        // The implementation matches my simulation, but the result doesn't create the shared prefix edge.
        // Let's rethink the algorithm description vs. the code.
        // The code moves `current_node` under `marker_node` *if* `current_label` starts with `marker_label`.
        //
        // Consider input: [(1, b("a")), (10, b("apple")), (20, b("apply"))]
        // Initial root.children: { b("a"): Node(1), b("apple"): Node(10), b("apply"): Node(20) }
        // merge_nodes(&root):
        //   marker = (b("a"), Node(1))
        //   peek = (b("apple"), _) -> b("apple").starts_with(b("a")) -> true
        //     consume (b("apple"), Node(10)). suffix = b("pple"). marker_node(1).children.insert(b("pple"), Node(10))
        //   peek = (b("apply"), _) -> b("apply").starts_with(b("a")) -> true
        //     consume (b("apply"), Node(20)). suffix = b("pply"). marker_node(1).children.insert(b("pply"), Node(20))
        //   peek = None. Break inner loop.
        //   merge_nodes(&mut marker_node(1)) -> will process children {b("pple"): Node(10), b("pply"): Node(20)}
        //     marker = (b("pple"), Node(10))
        //     peek = (b("pply"), _) -> b("pply").starts_with(b("pple")) -> false. Break inner loop.
        //     merge_nodes(&mut Node(10)) -> no-op
        //     new_children_for_a.insert(b("pple"), Node(10))
        //     marker = (b("pply"), Node(20))
        //     peek = None. End inner loop.
        //     merge_nodes(&mut Node(20)) -> no-op
        //     new_children_for_a.insert(b("pply"), Node(20))
        //     marker_node(1).children = {b("pple"): Node(10), b("pply"): Node(20)} // This is the state after recursion returns
        //   new_children_for_root.insert(b("a"), marker_node(1)) // marker_node(1) now has children
        //   iter.next() -> None. End outer loop.
        //   root.children = { b("a"): Node(1) { children: {b("pple"): Node(10), b("pply"): Node(20)} } }
        //
        // This structure seems correct according to the algorithm! The key is that the *marker node itself* must exist as a token.
        // So, for the original `test_shared_prefix_branching` with just "apple" and "apply", the algorithm *correctly*
        // produces two separate branches from the root because there's no common *token* prefix node to attach them to.
        // If we add "app" as a token, it should change.

        // Let's test the case where the common prefix IS a token.
        let tokens_with_prefix = vec![(5, b("app")), (10, b("apple")), (20, b("apply"))];
        let trie_with_prefix = VocabTrie::build(&tokens_with_prefix);

        // Expected: root --"app"--> Node(5) --"le"--> Node(10)
        //                         |
        //                         --"ly"--> Node(20)

        assert_eq!(trie_with_prefix.root.children.len(), 1); // Only "app" edge from root
        assert!(trie_with_prefix.root.children.contains_key(&b("app")));

        let node_app = trie_with_prefix.root.children.get(&b("app")).unwrap();
        assert_eq!(node_app.token_id, Some(5)); // Node for "app" has ID 5
        assert_eq!(node_app.children.len(), 2); // Node "app" has two children

        assert!(node_app.children.contains_key(&b("le"))); // Edge "le"
        let node_apple = node_app.children.get(&b("le")).unwrap();
        assert_eq!(node_apple.token_id, Some(10));
        assert!(node_apple.children.is_empty());

        assert!(node_app.children.contains_key(&b("ly"))); // Edge "ly"
        let node_apply = node_app.children.get(&b("ly")).unwrap();
        assert_eq!(node_apply.token_id, Some(20));
        assert!(node_apply.children.is_empty());

        assert_eq!(trie_with_prefix.find_token(&b("app")), Some(5));
        assert_eq!(trie_with_prefix.find_token(&b("apple")), Some(10));
        assert_eq!(trie_with_prefix.find_token(&b("apply")), Some(20));
        assert_eq!(trie_with_prefix.find_token(&b("appl")), None); // Intermediate path
        assert_eq!(trie_with_prefix.find_token(&b("ap")), None);
    }


    #[test]
    fn test_complex_case() {
        let tokens = vec![
            (1, b("a")),
            (2, b("b")),
            (10, b("ape")),
            (11, b("apple")),
            (12, b("apply")),
            (20, b("banana")),
        ];
        let trie = VocabTrie::build(&tokens);

        // Expected structure:
        // root --"a"------> Node(1) --"pe" --> Node(10)
        //    |                      --"pple"-> Node(11)
        //    |                      --"pply"-> Node(12)
        //    |
        //    --"b"------> Node(2) --"anana"-> Node(20)

        assert_eq!(trie.root.children.len(), 2); // "a", "b"
        assert!(trie.root.children.contains_key(&b("a")));
        assert!(trie.root.children.contains_key(&b("b")));

        // Check branch 'a'
        let node_a = trie.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, Some(1));
        assert_eq!(node_a.children.len(), 3); // "pe", "pple", "pply" relative to "a"
        assert!(node_a.children.contains_key(&b("pe")));
        assert!(node_a.children.contains_key(&b("pple")));
        assert!(node_a.children.contains_key(&b("pply")));

        assert_eq!(node_a.children.get(&b("pe")).unwrap().token_id, Some(10));
        assert!(node_a.children.get(&b("pe")).unwrap().children.is_empty());
        assert_eq!(node_a.children.get(&b("pple")).unwrap().token_id, Some(11));
        assert!(node_a.children.get(&b("pple")).unwrap().children.is_empty());
        assert_eq!(node_a.children.get(&b("pply")).unwrap().token_id, Some(12));
        assert!(node_a.children.get(&b("pply")).unwrap().children.is_empty());

        // Check branch 'b'
        let node_b = trie.root.children.get(&b("b")).unwrap();
        assert_eq!(node_b.token_id, Some(2));
        assert_eq!(node_b.children.len(), 1); // "anana" relative to "b"
        assert!(node_b.children.contains_key(&b("anana")));

        assert_eq!(node_b.children.get(&b("anana")).unwrap().token_id, Some(20));
        assert!(node_b.children.get(&b("anana")).unwrap().children.is_empty());

        // Test lookups
        assert_eq!(trie.find_token(&b("a")), Some(1));
        assert_eq!(trie.find_token(&b("ape")), Some(10));
        assert_eq!(trie.find_token(&b("apple")), Some(11));
        assert_eq!(trie.find_token(&b("apply")), Some(12));
        assert_eq!(trie.find_token(&b("b")), Some(2));
        assert_eq!(trie.find_token(&b("banana")), Some(20));
        assert_eq!(trie.find_token(&b("app")), None); // Not a token
        assert_eq!(trie.find_token(&b("ban")), None); // Not a token
        assert_eq!(trie.find_token(&b("c")), None);
    }

     #[test]
    fn test_duplicate_token_bytes() {
         // If duplicate bytes are provided, the BTreeMap insert will keep the last one
         // before merging. The merge logic might then discard one.
         // Let's see what happens with "a", "ab", "a" (duplicate)
        let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("a"))];
        let trie = VocabTrie::build(&tokens);

        // Initial state before merge: root.children = { b("a"): Node(3), b("ab"): Node(2) } (Node(1) overwritten)
        // Merge:
        // marker = (b("a"), Node(3))
        // peek = (b("ab"), Node(2)) -> b("ab").starts_with(b("a")) -> true
        //   consume (b("ab"), Node(2)). suffix = b("b"). marker_node(3).children.insert(b("b"), Node(2))
        // peek = None.
        // merge_nodes(&mut marker_node(3)) -> processes children {b("b"): Node(2)} -> no-op internal merge
        // new_children_for_root.insert(b("a"), marker_node(3))
        // root.children = { b("a"): Node(3) { children: {b("b"): Node(2)} } }

        assert_eq!(trie.root.children.len(), 1);
        let node_a = trie.root.children.get(&b("a")).unwrap();
        // The node for "a" should have the ID from the *last* occurrence in the input list
        assert_eq!(node_a.token_id, Some(3));
        assert_eq!(node_a.children.len(), 1);

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, Some(2));
        assert!(node_ab.children.is_empty());

        assert_eq!(trie.find_token(&b("a")), Some(3)); // Finds ID 3
        assert_eq!(trie.find_token(&b("ab")), Some(2));
    }
}