use std::collections::BTreeMap;
use std::fmt;
use std::cmp::Ordering;

use bitvec::prelude::*;
use std::collections::btree_map;

// Represents a node in the VocabPrefixTree
#[derive(PartialEq)] // Keep derived PartialEq for structural comparison in tests
pub struct VocabPrefixTreeNode {
    /// The token ID corresponding to the path from the root to this node.
    /// Every node represents a valid token endpoint.
    /// The root node has ID 0 by convention, unless overwritten by an empty string token.
    token_id: usize,
    /// The length of the byte sequence from the root to this node.
    prefix_length: usize,
    /// Children nodes, keyed by the byte vector representing the edge label.
    /// BTreeMap ensures edges are sorted lexicographically by byte vector.
    children: BTreeMap<Vec<u8>, VocabPrefixTreeNode>,
    /// Bit vector indicating all token IDs reachable from or including this node.
    /// The length is max_token_id + 1.
    reachable_token_ids: BitVec,
}

impl VocabPrefixTreeNode {
    /// Creates a new node representing a token endpoint.
    fn new(token_id: usize, prefix_length: usize) -> Self {
        VocabPrefixTreeNode {
            token_id,
            prefix_length,
            children: BTreeMap::new(),
            // Initialize empty; will be computed after tree structure is built.
            reachable_token_ids: BitVec::new(),
        }
    }

    /// Returns the token ID corresponding to the path from the root to this node.
    pub fn token_id(&self) -> usize {
        self.token_id
    }

    /// Returns the length of the prefix for this node.
    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    /// Returns an iterator over the children of this node.
    /// The iterator yields pairs of `(&Vec<u8>, &VocabPrefixTreeNode)`, representing the edge label and the child node.
    pub fn children(&self) -> btree_map::Iter<'_, Vec<u8>, VocabPrefixTreeNode> {
        self.children.iter()
    }

    /// Returns a bitvec representing the set of token IDs reachable from this node.
    pub fn reachable_token_ids(&self) -> &BitVec {
        &self.reachable_token_ids
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
        debug_struct.field("prefix_length", &self.prefix_length);

        // Summarize reachable_token_ids for brevity
        let reachable_summary = format!(
            "{} set bits (len {})",
            self.reachable_token_ids.count_ones(),
            self.reachable_token_ids.len()
        );
        debug_struct.field("reachable_token_ids", &reachable_summary);

        // Format children concisely using the helper.
        let children_summary: BTreeMap<String, &VocabPrefixTreeNode> = self
            .children()            .map(|(k, v)| (format_bytes(k), v))
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
    pub root: VocabPrefixTreeNode,
    /// Flag indicating if the empty string `""` was explicitly provided as a token.
    max_token_id: usize,
    /// The maximum token ID encountered during build, used for BitVec sizing.
    has_empty_string_token: bool,
}

impl VocabPrefixTree {
    /// Creates an empty VocabPrefixTree.
    /// The root node is assigned token ID 0 by convention, and it's marked
    /// as not representing an explicit empty string token initially.
    pub fn new() -> Self {
        VocabPrefixTree {
            // Root node represents the empty prefix (length 0), ID 0 by convention.
            root: VocabPrefixTreeNode::new(0, 0),
            // Initially, assume no empty string token is present.
            // Max ID is 0 initially, will be updated during build.
            max_token_id: 0,
            has_empty_string_token: false,
        }
    }

    /// Builds the VocabPrefixTree from a list of tokens.
    /// Tokens are provided as (token_id, byte_vector) pairs.
    /// The construction algorithm ensures that if token P is a prefix of token Q,
    /// the node for P becomes an ancestor of the node for Q.
    /// If an empty string token "" is provided, its ID overwrites the root's
    /// default ID 0, and the `has_empty_string_token` flag is set.
    pub fn build(tokens: &[(usize, Vec<u8>)]) -> Self {
        let mut tree = VocabPrefixTree::new(); // Root starts with ID 0, flag false

        // Determine the maximum token ID for BitVec sizing.
        // Handle empty input gracefully.
        tree.max_token_id = tokens.iter().map(|(id, _)| *id).max().unwrap_or(0);
                                               // Root prefix_length is 0
        // 1. Initial population: Add all tokens as direct children of the root.
        //    Each edge uses the full token byte vector as its label, leading
        //    to a leaf node holding the token's ID.
        crate::debug!(2, "Building vocab prefix tree");
        for (id, bytes) in tokens {
            crate::debug!(3, "Adding token {} with bytes {:?}", id, bytes);
            if bytes.is_empty() {
                // Assign the token ID for the empty string directly to the root,
                // overwriting the default 0 if necessary.
                 tree.root.token_id = *id;
                 // tree.root.prefix_length remains 0, which is correct.
                 // Mark that the root ID now represents an actual token.
                 tree.has_empty_string_token = true;
                continue;
            }
            // Insert node. If duplicate byte vecs exist, the last ID wins due to BTreeMap semantics.
            // The prefix_length is the length of the full token bytes.
            let node = VocabPrefixTreeNode::new(*id, bytes.len());
            tree.root.children.insert(bytes.clone(), node);
        }

        // 2. Merge nodes recursively starting from the root's children.
        //    This step restructures the tree into the compact radix form
        //    based on shared prefixes that are themselves valid tokens.
        crate::debug!(2, "Merging nodes");
        Self::merge_nodes(&mut tree.root);

        // 3. Compute reachable token IDs for all nodes in a post-order traversal.
        crate::debug!(2, "Computing reachable IDs");
        Self::compute_reachable_ids_recursive(&mut tree.root, tree.max_token_id);
        crate::debug!(2, "Done computing reachable IDs");

        // 4. Adjust root's reachable IDs if its ID 0 is just the convention.
        if !tree.has_empty_string_token && tree.root.token_id == 0 && tree.max_token_id >= 0 && !tree.root.reachable_token_ids.is_empty() {
            tree.root.reachable_token_ids.set(0, false);
        }

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

    /// Recursively computes the `reachable_token_ids` for each node.
    /// This should be called after the tree structure is finalized by `merge_nodes`.
    fn compute_reachable_ids_recursive(node: &mut VocabPrefixTreeNode, max_token_id: usize) {
        // Initialize the BitVec for the current node.
        let mut current_node_ids = bitvec![0; max_token_id + 1];

        // Set the bit for the node's own token ID, if valid.
        if node.token_id <= max_token_id {
            current_node_ids.set(node.token_id, true);
        }

        // Recursively call on children and merge their results.
        for child_node in node.children.values_mut() {
            Self::compute_reachable_ids_recursive(child_node, max_token_id);
            // OR the child's computed reachable IDs into the current node's set.
            current_node_ids |= &child_node.reachable_token_ids;
        }

        // Assign the final computed set to the node.
        node.reachable_token_ids = current_node_ids;
    }

     /// Finds the token ID corresponding to the exact byte sequence.
     /// Returns `None` if the sequence does not correspond to a token in the tree.
     /// Specifically, searching for the empty string `b""` only succeeds if an
     /// empty string token was explicitly added during the build process.
    pub fn find_token(&self, bytes: &[u8]) -> Option<usize> {
        if bytes.is_empty() {
            // Only return the root's ID if it represents an actual empty string token.
            if self.has_empty_string_token {
                return Some(self.root.token_id);
            } else {
                // The root ID (likely 0) is just a convention, not a real token here.
                return None;
            }
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
                // Return its token_id. Every node has one.
                return Some(current_node.token_id);
            }
            // If remaining_bytes is not empty, continue the loop from the new current_node.
        }
    }

    /// Returns `true` if the empty string `""` was provided as a token
    /// during the build process, `false` otherwise.
    pub fn has_empty_string_token(&self) -> bool {
        self.has_empty_string_token
    }

    /// Returns an iterator over the direct children of the root node.
    /// The iterator yields pairs of `(&Vec<u8>, &VocabPrefixTreeNode)`, representing the edge label and the child node.
    pub fn root_children(&self) -> btree_map::Iter<'_, Vec<u8>, VocabPrefixTreeNode> {
        self.root.children()
    }

    /// Returns the maximum token ID used to build this tree.
    pub fn max_token_id(&self) -> usize {
        self.max_token_id
    }
}

impl Default for VocabPrefixTree {
    fn default() -> Self {
        Self::new()
    }
}

// Need Eq to implement Ord
impl Eq for VocabPrefixTreeNode {}

impl PartialOrd for VocabPrefixTreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VocabPrefixTreeNode {
    /// Compares nodes primarily based on `prefix_length` and secondarily on `token_id`.
    /// Note: This ordering ignores the `children` field and is therefore NOT necessarily
    /// consistent with the derived `PartialEq` implementation which compares all fields.
    fn cmp(&self, other: &Self) -> Ordering {
        self.prefix_length.cmp(&other.prefix_length)
            .then_with(|| self.token_id.cmp(&other.token_id))
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    // Helper to create byte vecs from strings for tests.
    fn b(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn test_empty_tree() {
        let tokens: Vec<(usize, Vec<u8>)> = vec![];
        let tree = VocabPrefixTree::build(&tokens);
        assert_eq!(tree.root.token_id, 0); // Root defaults to 0
        assert_eq!(tree.root.prefix_length, 0); // Root length is 0
        assert!(!tree.has_empty_string_token()); // No empty token provided
        assert_eq!(tree.max_token_id(), 0); // Max ID is 0 for empty input
        assert!(tree.root.children.is_empty());
        assert_eq!(tree.find_token(b"a"), None);
        assert_eq!(tree.find_token(b""), None); // Empty query returns None (flag is false)
        // Root's reachable IDs should be empty (bit 0 removed as it's conventional)
        assert!(tree.root.reachable_token_ids.not_any());
    }

    #[test]
    fn test_single_token() {
        let tokens = vec![(1, b("hello"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 0); // Root ID remains 0
        assert_eq!(tree.root.prefix_length, 0);
        assert!(!tree.has_empty_string_token()); // No empty token provided
        assert_eq!(tree.max_token_id(), 1);
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("hello")));

        let node = tree.root.children.get(&b("hello")).unwrap();
        assert_eq!(node.token_id, 1);
        assert_eq!(node.prefix_length, 5); // "hello" has length 5
        assert!(node.children.is_empty());
        // Node "hello" should only have its own ID reachable
        let expected_node_bits = bitvec![0, 1]; // Size max_id + 1 = 2
        assert_eq!(node.reachable_token_ids, expected_node_bits);

        assert_eq!(tree.find_token(&b("hello")), Some(1));
        assert_eq!(tree.find_token(&b("hell")), None);
        assert_eq!(tree.find_token(&b("hello world")), None);
        assert_eq!(tree.find_token(b""), None); // Flag is false

        // Root's reachable IDs should contain only ID 1 (ID 0 is conventional)
        let expected_root_bits = bitvec![0, 1]; // Size max_id + 1 = 2
        assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
    }

     #[test]
    fn test_empty_string_token() {
        // Assign ID 99 to the empty string
        let tokens = vec![(99, b("")), (1, b("a"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 99); // Root gets the ID for ""
        assert_eq!(tree.root.prefix_length, 0); // Empty string has length 0
        assert!(tree.has_empty_string_token()); // Empty token WAS provided
        assert_eq!(tree.max_token_id(), 99);
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("a")));

        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 1);
        // Node "a" should only have its own ID reachable
        let mut expected_node_a_bits = bitvec![0; 100]; // Size max_id + 1 = 100
        expected_node_a_bits.set(1, true);
        assert_eq!(node_a.reachable_token_ids, expected_node_a_bits);

        assert_eq!(tree.find_token(&b("")), Some(99)); // Query for "" returns its ID (flag is true)
        assert_eq!(tree.find_token(&b("a")), Some(1));

        // Root's reachable IDs should contain 1 (from child) and 99 (itself)
        let mut expected_root_bits = bitvec![0; 100];
        expected_root_bits.set(1, true);
        expected_root_bits.set(99, true);
        assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
    }

    #[test]
    fn test_empty_string_token_with_id_zero() {
        // Assign ID 0 to the empty string explicitly
        let tokens = vec![(0, b("")), (1, b("a"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 0); // Root gets the ID 0 for ""
        assert_eq!(tree.root.prefix_length, 0);
        assert!(tree.has_empty_string_token()); // Empty token WAS provided
        assert_eq!(tree.max_token_id(), 1);
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("a")));

        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 1);
        // Node "a" should only have its own ID reachable
        let expected_node_a_bits = bitvec![0, 1]; // Size max_id + 1 = 2
        assert_eq!(node_a.reachable_token_ids, expected_node_a_bits);

        assert_eq!(tree.find_token(&b("")), Some(0)); // Query for "" returns its ID 0 (flag is true)
        assert_eq!(tree.find_token(&b("a")), Some(1));

        // Root's reachable IDs should contain 1 (from child) and 0 (itself, as it's explicit)
        let mut expected_root_bits = bitvec![0; 2];
        expected_root_bits.set(0, true);
        expected_root_bits.set(1, true);
        assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
    }


    #[test]
    fn test_simple_prefix() {
        // "a" is prefix of "ab"
        let tokens = vec![(1, b("a")), (2, b("ab"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 0);
        assert_eq!(tree.root.prefix_length, 0);
        assert!(!tree.has_empty_string_token());
        assert_eq!(tree.max_token_id(), 2);
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("a")));

        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 1);
        assert_eq!(node_a.prefix_length, 1); // "a" length 1
        assert_eq!(node_a.children.len(), 1);
        assert!(node_a.children.contains_key(&b("b")));

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.prefix_length, 2); // "ab" length 2
        assert_eq!(node_ab.token_id, 2);
        assert!(node_ab.children.is_empty());
        // Node "ab" reachable IDs: {2}
        let expected_ab_bits = bitvec![0, 0, 1]; // Size 3
        assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

        // Node "a" reachable IDs: {1, 2}
        let expected_a_bits = bitvec![0, 1, 1]; // Size 3
        assert_eq!(node_a.reachable_token_ids, expected_a_bits);

        assert_eq!(tree.find_token(&b("a")), Some(1));
        assert_eq!(tree.find_token(&b("ab")), Some(2));
        assert_eq!(tree.find_token(&b("b")), None);
        assert_eq!(tree.find_token(&b("abc")), None);
        assert_eq!(tree.find_token(b""), None); // Flag is false

        // Root reachable IDs: {1, 2} (0 is conventional)
        assert_eq!(tree.root.reachable_token_ids, expected_a_bits);
    }

    #[test]
    fn test_multiple_prefixes() {
        // "a", "ab", "abc"
        let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("abc"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 0);
        assert_eq!(tree.root.prefix_length, 0);
        assert!(!tree.has_empty_string_token());
        assert_eq!(tree.max_token_id(), 3);
        assert_eq!(tree.root.children.len(), 1);
        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 1);
        assert_eq!(node_a.prefix_length, 1);
        assert_eq!(node_a.children.len(), 1);

        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.token_id, 2);
        assert_eq!(node_ab.prefix_length, 2);
        assert_eq!(node_ab.children.len(), 1);

        let node_abc = node_ab.children.get(&b("c")).unwrap();
        assert_eq!(node_abc.prefix_length, 3);
        assert_eq!(node_abc.token_id, 3);
        assert!(node_abc.children.is_empty());
        // Node "abc" reachable: {3}
        let expected_abc_bits = bitvec![0, 0, 0, 1]; // Size 4
        assert_eq!(node_abc.reachable_token_ids, expected_abc_bits);

        // Node "ab" reachable: {2, 3}
        let expected_ab_bits = bitvec![0, 0, 1, 1]; // Size 4
        assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

        // Node "a" reachable: {1, 2, 3}
        let expected_a_bits = bitvec![0, 1, 1, 1]; // Size 4
        assert_eq!(node_a.reachable_token_ids, expected_a_bits);

        assert_eq!(tree.find_token(&b("a")), Some(1));
        assert_eq!(tree.find_token(&b("ab")), Some(2));
        assert_eq!(tree.find_token(&b("abc")), Some(3));
        assert_eq!(tree.find_token(&b("b")), None);
        assert_eq!(tree.find_token(&b("abcd")), None);

        // Root reachable: {1, 2, 3}
        assert_eq!(tree.root.reachable_token_ids, expected_a_bits);
        assert_eq!(tree.find_token(b""), None); // Flag is false
    }

    #[test]
    fn test_shared_prefix_branching_where_prefix_is_token() {
        let tokens_with_prefix = vec![(5, b("app")), (10, b("apple")), (20, b("apply"))];
        let tree_with_prefix = VocabPrefixTree::build(&tokens_with_prefix);

        assert_eq!(tree_with_prefix.root.token_id, 0);
        assert_eq!(tree_with_prefix.root.prefix_length, 0);
        assert!(!tree_with_prefix.has_empty_string_token());
        assert_eq!(tree_with_prefix.max_token_id(), 20);
        assert_eq!(tree_with_prefix.root.children.len(), 1);
        assert!(tree_with_prefix.root.children.contains_key(&b("app")));

        let node_app = tree_with_prefix.root.children.get(&b("app")).unwrap();
        assert_eq!(node_app.token_id, 5);
        assert_eq!(node_app.prefix_length, 3); // "app" length 3
        assert_eq!(node_app.children.len(), 2);

        assert!(node_app.children.contains_key(&b("le")));
        let node_apple = node_app.children.get(&b("le")).unwrap();
        assert_eq!(node_apple.token_id, 10);
        assert_eq!(node_apple.prefix_length, 5); // "apple" length 5
        assert!(node_apple.children.is_empty());
        // Node "apple" reachable: {10}
        let mut expected_apple_bits = bitvec![0; 21];
        expected_apple_bits.set(10, true);
        assert_eq!(node_apple.reachable_token_ids, expected_apple_bits);

        assert!(node_app.children.contains_key(&b("ly")));
        let node_apply = node_app.children.get(&b("ly")).unwrap();
        assert_eq!(node_apply.prefix_length, 5); // "apply" length 5
        assert_eq!(node_apply.token_id, 20);
        assert!(node_apply.children.is_empty());
        // Node "apply" reachable: {20}
        let mut expected_apply_bits = bitvec![0; 21];
        expected_apply_bits.set(20, true);
        assert_eq!(node_apply.reachable_token_ids, expected_apply_bits);

        // Node "app" reachable: {5, 10, 20}
        let mut expected_app_bits = bitvec![0; 21];
        expected_app_bits.set(5, true);
        expected_app_bits.set(10, true);
        expected_app_bits.set(20, true);
        assert_eq!(node_app.reachable_token_ids, expected_app_bits);

        assert_eq!(tree_with_prefix.find_token(&b("app")), Some(5));
        assert_eq!(tree_with_prefix.find_token(&b("apple")), Some(10));
        assert_eq!(tree_with_prefix.find_token(&b("apply")), Some(20));
        assert_eq!(tree_with_prefix.find_token(&b("appl")), None);
        assert_eq!(tree_with_prefix.find_token(&b("ap")), None);
        // Root reachable: {5, 10, 20}
        assert_eq!(tree_with_prefix.root.reachable_token_ids, expected_app_bits);
        assert_eq!(tree_with_prefix.find_token(b""), None); // Flag is false
    }

     #[test]
    fn test_shared_prefix_branching_where_prefix_is_not_token() {
        let tokens = vec![(10, b("apple")), (20, b("apply"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 0);
        assert_eq!(tree.root.prefix_length, 0);
        assert!(!tree.has_empty_string_token());
        assert_eq!(tree.max_token_id(), 20);
        assert_eq!(tree.root.children.len(), 2);
        assert!(tree.root.children.contains_key(&b("apple")));
        assert!(tree.root.children.contains_key(&b("apply")));

        let node_apple = tree.root.children.get(&b("apple")).unwrap();
        assert_eq!(node_apple.prefix_length, 5);
        assert_eq!(node_apple.token_id, 10);
        assert!(node_apple.children.is_empty());
        // Node "apple" reachable: {10}
        let mut expected_apple_bits = bitvec![0; 21];
        expected_apple_bits.set(10, true);
        assert_eq!(node_apple.reachable_token_ids, expected_apple_bits);

        let node_apply = tree.root.children.get(&b("apply")).unwrap();
        assert_eq!(node_apply.prefix_length, 5);
        assert_eq!(node_apply.token_id, 20);
        assert!(node_apply.children.is_empty());
        // Node "apply" reachable: {20}
        let mut expected_apply_bits = bitvec![0; 21];
        expected_apply_bits.set(20, true);
        assert_eq!(node_apply.reachable_token_ids, expected_apply_bits);

        assert_eq!(tree.find_token(&b("apple")), Some(10));
        assert_eq!(tree.find_token(&b("apply")), Some(20));
        assert_eq!(tree.find_token(&b("app")), None);
        assert_eq!(tree.find_token(&b("appl")), None);

        // Root reachable: {10, 20}
        let mut expected_root_bits = bitvec![0; 21];
        expected_root_bits.set(10, true);
        expected_root_bits.set(20, true);
        assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
        assert_eq!(tree.find_token(b""), None); // Flag is false
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
            (99, b("")), // Add empty token
        ];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 99); // Root ID set by "" token
        assert_eq!(tree.root.prefix_length, 0);
        assert!(tree.has_empty_string_token()); // Flag is true
        assert_eq!(tree.max_token_id(), 99);
        assert_eq!(tree.root.children.len(), 2);
        assert!(tree.root.children.contains_key(&b("a")));
        assert!(tree.root.children.contains_key(&b("b")));

        // Check branch 'a'
        // The merge logic places nodes whose token is prefixed by another token
        // as children of that prefix token's node.
        // "a" (1) is a token. "ape"(10), "apple"(11), "apply"(12) all start with "a".
        // So, they become children of the node for "a".
        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 1);
        assert_eq!(node_a.prefix_length, 1);
        assert_eq!(node_a.children.len(), 3);
        assert!(node_a.children.contains_key(&b("pe")));
        assert!(node_a.children.contains_key(&b("pple")));
        assert!(node_a.children.contains_key(&b("pply")));
        let node_ape = node_a.children.get(&b("pe")).unwrap();
        let node_apple = node_a.children.get(&b("pple")).unwrap();
        let node_apply = node_a.children.get(&b("pply")).unwrap();
        assert_eq!(node_ape.token_id, 10);
        assert_eq!(node_ape.prefix_length, 3); // "ape"
        assert_eq!(node_apple.token_id, 11);
        assert_eq!(node_apple.prefix_length, 5); // "apple"
        assert_eq!(node_apply.token_id, 12);
        assert_eq!(node_apply.prefix_length, 5); // "apply"

        // Node "a" reachable: {1, 10, 11, 12}
        let mut expected_a_bits = bitvec![0; 100];
        expected_a_bits.set(1, true);
        expected_a_bits.set(10, true);
        expected_a_bits.set(11, true);
        expected_a_bits.set(12, true);
        assert_eq!(node_a.reachable_token_ids, expected_a_bits);


        // Check branch 'b'
        let node_b = tree.root.children.get(&b("b")).unwrap();
        assert_eq!(node_b.token_id, 2);
        assert_eq!(node_b.prefix_length, 1);
        assert_eq!(node_b.children.len(), 1);
        assert!(node_b.children.contains_key(&b("anana")));
        assert_eq!(node_b.children.get(&b("anana")).unwrap().token_id, 20);
        let node_banana = node_b.children.get(&b("anana")).unwrap();
        assert_eq!(node_banana.prefix_length, 6); // "banana"

        // Node "b" reachable: {2, 20}
        let mut expected_b_bits = bitvec![0; 100];
        expected_b_bits.set(2, true);
        expected_b_bits.set(20, true);
        assert_eq!(node_b.reachable_token_ids, expected_b_bits);

        // Test lookups
        assert_eq!(tree.find_token(&b("a")), Some(1));
        assert_eq!(tree.find_token(&b("ape")), Some(10));
        assert_eq!(tree.find_token(&b("apple")), Some(11));
        assert_eq!(tree.find_token(&b("apply")), Some(12));
        assert_eq!(tree.find_token(&b("b")), Some(2));
        assert_eq!(tree.find_token(&b("banana")), Some(20));
        assert_eq!(tree.find_token(&b("app")), None);
        assert_eq!(tree.find_token(&b("ban")), None);
        assert_eq!(tree.find_token(&b("c")), None);

        // Root reachable: {1, 2, 10, 11, 12, 20, 99}
        let mut expected_root_bits = bitvec![0; 100];
        expected_root_bits.set(1, true);
        expected_root_bits.set(2, true);
        expected_root_bits.set(10, true);
        expected_root_bits.set(11, true);
        expected_root_bits.set(12, true);
        expected_root_bits.set(20, true);
        expected_root_bits.set(99, true);
        assert_eq!(tree.find_token(b""), Some(99)); // Query for "" returns its ID (flag is true)
    }

     #[test]
    fn test_duplicate_token_bytes() {
        let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("a"))];
        let tree = VocabPrefixTree::build(&tokens);

        assert_eq!(tree.root.token_id, 0);
        assert_eq!(tree.root.prefix_length, 0);
        assert!(!tree.has_empty_string_token());
        assert_eq!(tree.max_token_id(), 3);
        assert_eq!(tree.root.children.len(), 1);
        assert!(tree.root.children.contains_key(&b("a")));

        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 3); // Last ID wins
        assert_eq!(node_a.prefix_length, 1); // Length of "a"
        assert_eq!(node_a.children.len(), 1);

        assert!(node_a.children.contains_key(&b("b")));
        let node_ab = node_a.children.get(&b("b")).unwrap();
        assert_eq!(node_ab.prefix_length, 2); // Length of "ab"
        assert_eq!(node_ab.token_id, 2);
        assert!(node_ab.children.is_empty());
        // Node "ab" reachable: {2}
        let expected_ab_bits = bitvec![0, 0, 1, 0]; // Size 4
        assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

        // Node "a" reachable: {2, 3} (ID 1 was overwritten)
        let expected_a_bits = bitvec![0, 0, 1, 1]; // Size 4
        assert_eq!(node_a.reachable_token_ids, expected_a_bits);

        assert_eq!(tree.find_token(&b("a")), Some(3));
        assert_eq!(tree.find_token(&b("ab")), Some(2));
        // Root reachable: {2, 3}
        assert_eq!(tree.root.reachable_token_ids, expected_a_bits);
        assert_eq!(tree.find_token(b""), None); // Flag is false
    }

    #[test]
    fn test_children_iteration() {
        let tokens = vec![
            (1, b("a")),
            (2, b("b")),
            (10, b("ape")),
            (20, b("banana")),
        ];
        let tree = VocabPrefixTree::build(&tokens);

        // Iterate root children
        let mut root_children_iter = tree.root_children();

        let (edge_a, node_a_ref) = root_children_iter.next().unwrap();
        assert_eq!(edge_a, &b("a"));
        assert_eq!(node_a_ref.token_id, 1);
        assert_eq!(node_a_ref.prefix_length, 1);

        let (edge_b, node_b_ref) = root_children_iter.next().unwrap();
        assert_eq!(edge_b, &b("b"));
        assert_eq!(node_b_ref.token_id, 2);
        assert_eq!(node_b_ref.prefix_length, 1);

        assert!(root_children_iter.next().is_none()); // Only 'a' and 'b' are direct children of root

        // Iterate children of node 'a'
        let mut node_a_children_iter = node_a_ref.children();
        let (edge_pe, node_ape_ref) = node_a_children_iter.next().unwrap();
        assert_eq!(edge_pe, &b("pe"));
        assert_eq!(node_ape_ref.token_id, 10);
        assert_eq!(node_ape_ref.prefix_length, 3); // "ape"
        assert!(node_a_children_iter.next().is_none());

        // Iterate children of node 'b'
        let mut node_b_children_iter = node_b_ref.children();
        let (edge_anana, node_banana_ref) = node_b_children_iter.next().unwrap();
        assert_eq!(edge_anana, &b("anana"));
        assert_eq!(node_banana_ref.token_id, 20);
        assert_eq!(node_banana_ref.prefix_length, 6); // "banana"
        assert!(node_b_children_iter.next().is_none());
    }
}

