use std::collections::BTreeMap;
use std::fmt;

// Represents a node in the VocabPrefixTree
#[derive(PartialEq)]
pub struct VocabPrefixTreeNode {
    /// The token ID if the path from the root to this node represents a complete token.
    /// The root node will have None unless the empty string is a token.
    token_id: Option<u32>,
    /// Children nodes, keyed by the byte vector representing the edge label.
    /// BTreeMap ensures edges are sorted lexicographically by byte vector,
    /// which is important for the deterministic construction algorithm.
    children: BTreeMap<Vec<u8>, VocabPrefixTreeNode>,
}

impl VocabPrefixTreeNode {
    /// Creates a new node, optionally representing a token endpoint.
    fn new(token_id: Option<u32>) -> Self {
        VocabPrefixTreeNode {
            token_id,
            children: BTreeMap::new(),
        }
    }
}

// Manual implementation of Debug for concise node representation.
impl fmt::Debug for VocabPrefixTreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Helper to format byte vectors for display, limiting length.
        fn format_bytes(bytes: &[u8]) -> String {
            const MAX_BYTES_DISPLAY: usize = 10;
            let display_str = String::from_utf8_lossy(bytes.get(..MAX_BYTES_DISPLAY).unwrap_or(bytes));
            if bytes.len() > MAX_BYTES_DISPLAY {
                format!("{}...({} bytes)", display_str, bytes.len())
            } else {
                format!("{}", display_str)
            }
        }

        let mut debug_struct = f.debug_struct("VocabPrefixTreeNode");
        debug_struct.field("token_id", &self.token_id);

        // Format children concisely using the helper.
        let children_summary: BTreeMap<String, &VocabPrefixTreeNode> = self
            .children
            .iter()
            .map(|(k, v)| (format_bytes(k), v))
            .collect();
        debug_struct.field("children", &children_summary);

        debug_struct.finish()
    }
}


/// A vocabulary prefix tree (a specialized radix tree) where every node
/// corresponds to a valid token ID from the input vocabulary.
/// Edges are labeled with byte vectors.
#[derive(Debug, PartialEq)]
pub struct VocabPrefixTree {
    root: VocabPrefixTreeNode,
}

impl VocabPrefixTree {
    /// Creates an empty VocabPrefixTree.
    pub fn new() -> Self {
        VocabPrefixTree {
            // Root node represents the empty prefix.
            root: VocabPrefixTreeNode::new(None),
        }
    }

    /// Builds the VocabPrefixTree from a list of tokens.
    /// Tokens are provided as (token_id, byte_vector) pairs.
    /// The construction algorithm ensures that if token P is a prefix of token Q,
    /// the node for P becomes an ancestor of the node for Q.
    pub fn build(tokens: &[(u32, Vec<u8>)]) -> Self {
        let mut tree = VocabPrefixTree::new();

        // 1. Initial population: Add all tokens as direct children of the root.
        //    Each edge uses the full token byte vector as its label, leading
        //    to a leaf node holding the token's ID.
        for (id, bytes) in tokens {
            if bytes.is_empty() {
                // Assign the token ID for the empty string directly to the root.
                 tree.root.token_id = Some(*id);
                continue;
            }
            // Insert node. If duplicate byte vecs exist, the last ID wins due to BTreeMap semantics.
            tree.root
                .children
                .insert(bytes.clone(), VocabPrefixTreeNode::new(Some(*id)));
        }

        // 2. Merge nodes recursively starting from the root's children.
        //    This step restructures the tree into the compact radix form
        //    based on shared prefixes that are themselves valid tokens.
        Self::merge_nodes(&mut tree.root);

        tree
    }

    /// Recursively merges nodes based on the prefix relationship described.
    /// Assumes `node.children` is sorted lexicographically (guaranteed by BTreeMap).
    fn merge_nodes(node: &mut VocabPrefixTreeNode) {
        if node.children.len() <= 1 {
            // Base case: No merging needed if 0 or 1 child.
            // Still need to recurse down in case the single child has children needing merging.
            for child_node in node.children.values_mut() {
                Self::merge_nodes(child_node);
            }
            return;
        }

        // Take ownership of the children map to rebuild it during the merge process.
        let old_children = std::mem::take(&mut node.children);
        let mut new_children = BTreeMap::new();

        // Use an iterator to process children in sorted byte order.
        let mut iter = old_children.into_iter().peekable();

        while let Some((marker_label, mut marker_node)) = iter.next() {
            // `marker_node` corresponds to the token `marker_label`.
            // Check subsequent nodes to see if they should become children of `marker_node`.

            while let Some((current_label, _)) = iter.peek() {
                // Check if `current_label` starts with `marker_label`.
                if current_label.starts_with(&marker_label) {
                    // Yes, the token `current_label` has `marker_label` as a prefix.
                    // Consume the current item from the iterator.
                    let (current_label_owned, current_node) = iter.next().unwrap();

                    // Calculate the suffix: the part of `current_label_owned` after `marker_label`.
                    let suffix = current_label_owned[marker_label.len()..].to_vec();

                    // Add `current_node` as a child of `marker_node` using the suffix as the edge label.
                    if !suffix.is_empty() {
                         marker_node.children.insert(suffix, current_node);
                    } else {
                        // This case implies current_label == marker_label (duplicate token bytes).
                        // The BTreeMap insertion during initial population already handled this
                        // by keeping the last ID. The `marker_node` already represents this token.
                        // We effectively discard the duplicate node structure here.
                        // Log a warning as this might indicate an issue in the input vocabulary.
                         eprintln!("Warning: Duplicate token bytes encountered and merged: {:?}", marker_label);
                    }

                } else {
                    // No prefix match, this node starts a new group relative to the current parent `node`.
                    // Stop checking against the current `marker_node`.
                    break;
                }
            }

            // After potentially adding children to `marker_node`, recursively merge *its* new children.
            // This ensures the prefix structure propagates down the tree.
            Self::merge_nodes(&mut marker_node);

            // Add the (potentially updated) `marker_node` back into the parent's `new_children` map.
            new_children.insert(marker_label, marker_node);
        }

        // Replace the original node's children with the newly structured map.
        node.children = new_children;
    }

     /// Finds the token ID corresponding to the exact byte sequence.
     /// Traverses the tree following matching edge prefixes.
    pub fn find_token(&self, bytes: &[u8]) -> Option<u32> {
        if bytes.is_empty() {
            // Handle lookup for the empty string token potentially stored at the root.
            return self.root.token_id;
        }

        let mut current_node = &self.root;
        let mut remaining_bytes = bytes;

        loop {
            let mut found_match = false;
            // Iterate through the children of the current node.
            for (edge_label, child_node) in &current_node.children {
                if remaining_bytes.starts_with(edge_label) {
                    // Found an edge matching a prefix of the remaining bytes.
                    remaining_bytes = &remaining_bytes[edge_label.len()..];
                    current_node = child_node; // Move down to the child node.
                    found_match = true;
                    break; // Proceed to the next level or check for final match.
                }
            }

            if !found_match {
                // No child edge matches the start of the remaining bytes.
                // The full sequence is not present in the tree.
                return None;
            }

            if remaining_bytes.is_empty() {
                // We have consumed all bytes and landed exactly on `current_node`.
                // Return its token_id. This will be Some(id) if this path corresponds
                // to a complete token, and None if it's only an intermediate path
                // (though the construction algorithm ensures nodes only exist for valid tokens).
                return current_node.token_id;
            }
            // If remaining_bytes is not empty, continue the loop from the new current_node.
        }
    }
}

impl Default for VocabPrefixTree {
    fn default() -> Self {
        Self::new()
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create byte vecs from strings for tests.
    fn b(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn test_empty_tree() {
        let tokens: Vec<(u32, Vec<u8>)> = vec![];
        let tree = VocabPrefixTree::build(&tokens);
        assert_eq!(tree.root.token_id, None);
        assert!(tree.root.children.is_empty());
        assert_eq!(tree.find_token(b"a"), None);
    }

    #[test]
    fn test_single_token() {
        let tokens = vec![(1, b("hello"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, None);
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("hello")));

        let node = tree.root.children.get(&b("hello")).unwrap();
        assert_eq!(node.token_id, Some(1));
        assert!(node.children.is_empty());

        assert_eq!(tree.find_token(&b("hello")), Some(1));
        assert_eq!(tree.find_token(&b("hell")), None);
        assert_eq!(tree.find_token(&b("hello world")), None);
    }

     #[test]
    fn test_empty_string_token() {
        let tokens = vec![(0, b("")), (1, b("a"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, Some(0)); // Root gets the ID for ""
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("a")));
        assert_eq!(tree.root.children[&b("a")].token_id, Some(1));

        assert_eq!(tree.find_token(&b("")), Some(0));
        assert_eq!(tree.find_token(&b("a")), Some(1));
    }

    #[test]
    fn test_simple_prefix() {
        // "a" is prefix of "ab"
        let tokens = vec![(1, b("a")), (2, b("ab"))];
        let tree = VocabPrefixTree::build(&tokens);

        // Expected structure: root --"a"--> Node(1) --"b"--> Node(2)
        assert_eq!(tree.root.children.len(), 1); // Only "a" edge from root
        assert!(tree.root.children.contains_key(&b("a")));

        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, Some(1)); // Node for "a" has ID 1
        assert_eq!(node_a.children.len(), 1); // Node "a" has one child
        assert!(node_a.children.contains_key(&b("b"))); // Edge is the suffix "b"

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, Some(2)); // Node for "ab" has ID 2
        assert!(node_ab.children.is_empty());

        assert_eq!(tree.find_token(&b("a")), Some(1));
        assert_eq!(tree.find_token(&b("ab")), Some(2));
        assert_eq!(tree.find_token(&b("b")), None);
        assert_eq!(tree.find_token(&b("abc")), None);
    }

    #[test]
    fn test_multiple_prefixes() {
        // "a", "ab", "abc"
        let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("abc"))];
        let tree = VocabPrefixTree::build(&tokens);

        // Expected: root --"a"--> Node(1) --"b"--> Node(2) --"c"--> Node(3)
        assert_eq!(tree.root.children.len(), 1);
        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, Some(1));
        assert_eq!(node_a.children.len(), 1);

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, Some(2));
        assert_eq!(node_ab.children.len(), 1);

        let node_abc = node_ab.children.get(&b("c")).unwrap();
        assert_eq!(node_abc.token_id, Some(3));
        assert!(node_abc.children.is_empty());

        assert_eq!(tree.find_token(&b("a")), Some(1));
        assert_eq!(tree.find_token(&b("ab")), Some(2));
        assert_eq!(tree.find_token(&b("abc")), Some(3));
        assert_eq!(tree.find_token(&b("b")), None);
        assert_eq!(tree.find_token(&b("abcd")), None);
    }

    #[test]
    fn test_shared_prefix_branching_where_prefix_is_token() {
        // Test case: "app", "apple", "apply"
        // "app" is a token and a prefix of the others.
        let tokens_with_prefix = vec![(5, b("app")), (10, b("apple")), (20, b("apply"))];
        let tree_with_prefix = VocabPrefixTree::build(&tokens_with_prefix);

        // Expected: root --"app"--> Node(5) --"le"--> Node(10)
        //                         |
        //                         --"ly"--> Node(20)

        assert_eq!(tree_with_prefix.root.children.len(), 1); // Only "app" edge from root
        assert!(tree_with_prefix.root.children.contains_key(&b("app")));

        let node_app = tree_with_prefix.root.children.get(&b("app")).unwrap();
        assert_eq!(node_app.token_id, Some(5)); // Node for "app" has ID 5
        assert_eq!(node_app.children.len(), 2); // Node "app" has two children ("le", "ly")

        assert!(node_app.children.contains_key(&b("le"))); // Edge "le"
        let node_apple = node_app.children.get(&b("le")).unwrap();
        assert_eq!(node_apple.token_id, Some(10));
        assert!(node_apple.children.is_empty());

        assert!(node_app.children.contains_key(&b("ly"))); // Edge "ly"
        let node_apply = node_app.children.get(&b("ly")).unwrap();
        assert_eq!(node_apply.token_id, Some(20));
        assert!(node_apply.children.is_empty());

        assert_eq!(tree_with_prefix.find_token(&b("app")), Some(5));
        assert_eq!(tree_with_prefix.find_token(&b("apple")), Some(10));
        assert_eq!(tree_with_prefix.find_token(&b("apply")), Some(20));
        assert_eq!(tree_with_prefix.find_token(&b("appl")), None); // Intermediate path, not a token
        assert_eq!(tree_with_prefix.find_token(&b("ap")), None);
    }

     #[test]
    fn test_shared_prefix_branching_where_prefix_is_not_token() {
        // Test case: "apple", "apply"
        // "appl" is a shared prefix but not a token itself.
        // The algorithm should create separate branches from the longest common *token* prefix,
        // which in this case is the root (empty string).
        let tokens = vec![(10, b("apple")), (20, b("apply"))];
        let tree = VocabPrefixTree::build(&tokens);

        // Expected: root --"apple"--> Node(10)
        //            |
        //            --"apply"--> Node(20)
        // Because "app" or "appl" are not tokens, they don't form intermediate nodes.
        assert_eq!(tree.root.children.len(), 2);
        assert!(tree.root.children.contains_key(&b("apple")));
        assert!(tree.root.children.contains_key(&b("apply")));

        let node_apple = tree.root.children.get(&b("apple")).unwrap();
        assert_eq!(node_apple.token_id, Some(10));
        assert!(node_apple.children.is_empty());

        let node_apply = tree.root.children.get(&b("apply")).unwrap();
        assert_eq!(node_apply.token_id, Some(20));
        assert!(node_apply.children.is_empty());

        assert_eq!(tree.find_token(&b("apple")), Some(10));
        assert_eq!(tree.find_token(&b("apply")), Some(20));
        assert_eq!(tree.find_token(&b("app")), None);
        assert_eq!(tree.find_token(&b("appl")), None);
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
        let tree = VocabPrefixTree::build(&tokens);

        // Expected structure based on the algorithm (prefixes must be tokens):
        // root --"a"------> Node(1) --"pe" --> Node(10)
        //    |                      --"pple"-> Node(11)
        //    |                      --"pply"-> Node(12)
        //    |
        //    --"b"------> Node(2) --"anana"-> Node(20)

        assert_eq!(tree.root.children.len(), 2); // Edges "a", "b" from root
        assert!(tree.root.children.contains_key(&b("a")));
        assert!(tree.root.children.contains_key(&b("b")));

        // Check branch 'a'
        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, Some(1));
        assert_eq!(node_a.children.len(), 3); // Edges "pe", "pple", "pply" relative to "a"
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
        let node_b = tree.root.children.get(&b("b")).unwrap();
        assert_eq!(node_b.token_id, Some(2));
        assert_eq!(node_b.children.len(), 1); // Edge "anana" relative to "b"
        assert!(node_b.children.contains_key(&b("anana")));

        let node_banana = node_b.children.get(&b("anana")).unwrap();
        assert_eq!(node_banana.token_id, Some(20));
        assert!(node_banana.children.is_empty());

        // Test lookups
        assert_eq!(tree.find_token(&b("a")), Some(1));
        assert_eq!(tree.find_token(&b("ape")), Some(10));
        assert_eq!(tree.find_token(&b("apple")), Some(11));
        assert_eq!(tree.find_token(&b("apply")), Some(12));
        assert_eq!(tree.find_token(&b("b")), Some(2));
        assert_eq!(tree.find_token(&b("banana")), Some(20));
        assert_eq!(tree.find_token(&b("app")), None); // Not a token
        assert_eq!(tree.find_token(&b("ban")), None); // Not a token
        assert_eq!(tree.find_token(&b("c")), None);
    }

     #[test]
    fn test_duplicate_token_bytes() {
         // Input: [(1, b("a")), (2, b("ab")), (3, b("a"))]
         // The last entry for "a" (ID 3) should overwrite the first one during initial population.
         // The merge process should then attach "ab" (ID 2) under the node for "a" (ID 3).
        let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("a"))];
        let tree = VocabPrefixTree::build(&tokens);

        // Expected: root --"a"--> Node(3) --"b"--> Node(2)
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("a")));

        let node_a = tree.root.children.get(&b("a")).unwrap();
        // The node for "a" should have the ID from the *last* occurrence in the input list.
        assert_eq!(node_a.token_id, Some(3));
        assert_eq!(node_a.children.len(), 1); // Should have child "b"

        assert!(node_a.children.contains_key(&b("b")));
        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, Some(2));
        assert!(node_ab.children.is_empty());

        assert_eq!(tree.find_token(&b("a")), Some(3)); // Finds ID 3
        assert_eq!(tree.find_token(&b("ab")), Some(2));
    }
}