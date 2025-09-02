use std::collections::BTreeMap;
use std::fmt;
use std::cmp::Ordering;
use std::collections::HashSet;

use bitvec::prelude::*; // Keep for macros or other uses if needed
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap; // Added for derive macro pattern


// Represents a node in the VocabPrefixTree
#[derive(PartialEq)] // Keep derived PartialEq for structural comparison in tests
pub struct VocabPrefixTreeNode {
    /// The token ID corresponding to the path from the root to this node.
    /// Every node represents a valid token endpoint.
    token_id: usize,
    /// The byte sequence from the root to this node (full prefix).
    prefix: Vec<u8>,
    /// The length of the byte sequence from the root to this node.
    prefix_length: usize,
    /// Children nodes, keyed by the byte vector representing the edge label.
    /// BTreeMap ensures edges are sorted lexicographically by byte vector.
    children: BTreeMap<Vec<u8>, VocabPrefixTreeNode>,
    /// Bit vector indicating all token IDs reachable from or including this node.
    /// The length is max_token_id + 1.
    reachable_token_ids: HybridBitset,
}

impl JSONConvertible for VocabPrefixTreeNode {
    fn to_json(&self) -> JSONNode {
        // WARNING: This is a naive recursive serialization.
        // For deep trees, it can lead to stack overflow or very large JSON.
        // A more robust solution might involve flattening or iterative approaches.
        todo!("VocabPrefixTreeNode to_json: Complex recursive structure.")
    }

    fn from_json(_node: JSONNode) -> Result<Self, String> {
        todo!("VocabPrefixTreeNode from_json: Complex recursive structure.")
    }
}


impl VocabPrefixTreeNode {
    /// Creates a new node representing a token endpoint.
    fn new(token_id: usize, prefix: Vec<u8>) -> Self {
        let prefix_length = prefix.len();
        VocabPrefixTreeNode {
            token_id,
            prefix,
            prefix_length,
            children: BTreeMap::new(),
            // Initialize empty; will be computed after tree structure is built.
            reachable_token_ids: HybridBitset::zeros(),
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

    /// Returns the prefix byte sequence for this node.
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    pub fn children(&self) -> &BTreeMap<Vec<u8>, VocabPrefixTreeNode> {
        &self.children
    }

    /// Returns an iterator over the children of this node.
    /// The iterator yields pairs of `(&Vec<u8>, &VocabPrefixTreeNode)`, representing the edge label and the child node.
    pub fn iter_children(&self) -> std::collections::btree_map::Iter<'_, Vec<u8>, VocabPrefixTreeNode> {
        self.children.iter()
    }

    /// Returns a bitset representing the set of token IDs reachable from this node (including the token this node itself represents).
    pub fn reachable_token_ids(&self) -> &HybridBitset {
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
            "{} set bits",
            self.reachable_token_ids.len() // Use len() for count
        );
        debug_struct.field("reachable_token_ids", &reachable_summary);

        // Format children concisely using the helper.
        let children_summary: BTreeMap<String, String> = self // Changed to BTreeMap<String, String> for Debug
            .iter_children()            
            .map(|(k, _v)| (format_bytes(k), format!("<VocabPrefixTreeNode ...>"))) // Don't recurse in Debug
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

impl JSONConvertible for VocabPrefixTree {
    fn to_json(&self) -> JSONNode {
        todo!("VocabPrefixTree to_json: Depends on VocabPrefixTreeNode.")
    }
    fn from_json(_node: JSONNode) -> Result<Self, String> {
        todo!("VocabPrefixTree from_json: Depends on VocabPrefixTreeNode.")
    }
}


impl VocabPrefixTree {
    /// Creates an empty VocabPrefixTree.
    /// The root node is assigned token ID 0 by convention, and it's marked
    /// as not representing an explicit empty string token initially.
    pub fn new() -> Self {
        VocabPrefixTree {
            // Root node represents the empty prefix (length 0), ID 0 by convention.
            root: VocabPrefixTreeNode::new(0, Vec::new()),
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
            crate::debug!(5, "Adding token {} with bytes {:?}", id, bytes);
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
            let node = VocabPrefixTreeNode::new(*id, bytes.clone());
            tree.root.children.insert(bytes.clone(), node);
        }

        // 2. Merge nodes recursively starting from the root's children.
        //    This step restructures the tree into the compact radix form
        //    based on shared prefixes that are themselves valid tokens.
        crate::debug!(2, "Merging nodes");
        Self::merge_nodes(&mut tree.root);

        // 3. Compute reachable token IDs for all nodes efficiently.
        //    We avoid building/merging large HashSets by distributing each node's token_id
        //    along its ancestor chain (including itself). This is O(total number of
        //    ancestor links across all tokens), which is typically far smaller than
        //    repeatedly unioning large sets.
        crate::debug!(2, "Computing reachable IDs (fast path)");
        tree.recompute_reachable_ids_via_paths();
        crate::debug!(2, "Done computing reachable IDs");

        // 4. Adjust root's reachable IDs if its ID 0 is just the convention.
        if !tree.has_empty_string_token && tree.root.token_id == 0 && !tree.root.reachable_token_ids.is_empty() {
            tree.root.reachable_token_ids.remove(0);
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

    // -------------- New, faster reachable IDs computation --------------

    /// Recomputes reachable_token_ids for all nodes using a fast, path-based propagation:
    /// For every token node in the tree, insert its token_id into the reachable bitset of
    /// every ancestor along the path from the root (including itself).
    fn recompute_reachable_ids_via_paths(&mut self) {
        // 1) Clear all reachable sets
        Self::clear_reachable_ids_recursive(&mut self.root);

        // 2) Collect all tokens present in the final, merged tree.
        //    If the empty string wasn't provided, skip the root (its token_id is conventional).
        let mut tokens_in_tree: Vec<(usize, Vec<u8>)> = Vec::new();
        Self::collect_tokens_recursive(
            &self.root,
            &mut tokens_in_tree,
            self.has_empty_string_token, // include root iff empty string token exists
        );

        // 3) For each token, find the sequence of edge labels (keys) along the path from
        //    the root to the token node, then insert that token_id into the reachable
        //    sets along that path in a single mutable pass.
        for (token_id, full_prefix) in tokens_in_tree {
            // Determine the path as a sequence of edge labels (Vec<u8>) from root to target.
            let path_keys = Self::compute_path_keys(&self.root, &full_prefix);

            // Mutably walk the path and insert the token_id into all visited nodes.
            Self::insert_token_along_path(&mut self.root, token_id, &path_keys);
        }
    }

    /// Clears reachable_token_ids for the entire subtree rooted at `node`.
    fn clear_reachable_ids_recursive(node: &mut VocabPrefixTreeNode) {
        node.reachable_token_ids = HybridBitset::zeros();
        for child in node.children.values_mut() {
            Self::clear_reachable_ids_recursive(child);
        }
    }

    /// Collects (token_id, full_prefix) pairs for all nodes in the subtree.
    /// If `include_this` is false, the current node is skipped (used to exclude root
    /// when no empty string token was provided).
    fn collect_tokens_recursive(
        node: &VocabPrefixTreeNode,
        out: &mut Vec<(usize, Vec<u8>)>,
        include_this: bool,
    ) {
        if include_this {
            out.push((node.token_id, node.prefix.clone()));
        }
        for child in node.children.values() {
            // Children are always included
            Self::collect_tokens_recursive(child, out, true);
        }
    }

    /// Given the root and a token's full prefix (node.prefix), compute the ordered list
    /// of edge labels (keys) from the root down to that node.
    fn compute_path_keys<'a>(
        mut current: &'a VocabPrefixTreeNode,
        target_prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();
        if target_prefix.is_empty() {
            // Empty string token: no edges to traverse
            return keys;
        }

        let mut remaining = target_prefix;
        'outer: loop {
            // Find the child edge that matches the current remaining bytes prefix
            for (edge_label, child) in current.children.iter() {
                if remaining.starts_with(edge_label.as_slice()) {
                    keys.push(edge_label.clone());
                    if remaining.len() == edge_label.len() {
                        // Reached the node exactly
                        break 'outer;
                    }
                    // Descend
                    current = child;
                    remaining = &remaining[edge_label.len()..];
                    // Continue searching at the new level
                    continue 'outer;
                }
            }
            // If we get here, something is inconsistent in the tree construction.
            // For robustness, break (no keys returned).
            break;
        }

        keys
    }

    /// Mutably traverses the tree following `path_keys` and inserts `token_id` into
    /// the reachable bitset of every node on the path (root and all descendants visited).
    fn insert_token_along_path(
        mut current: &mut VocabPrefixTreeNode,
        token_id: usize,
        path_keys: &[Vec<u8>],
    ) {
        // Root always accumulates all tokens
        current.reachable_token_ids.insert(token_id);

        for key in path_keys {
            // Temporarily descend into the child
            let child = current.children.get_mut(key).expect("Path key not found during reachable propagation.");
            // Insert for the child
            child.reachable_token_ids.insert(token_id);
            // Move down for the next iteration
            current = child;
        }
    }

    /// Returns `true` if the empty string `""` was provided as a token
    /// during the build process, `false` otherwise.
    pub fn has_empty_string_token(&self) -> bool {
        self.has_empty_string_token
    }

    /// Returns an iterator over the direct children of the root node.
    /// The iterator yields pairs of `(&Vec<u8>, &VocabPrefixTreeNode)`, representing the edge label and the child node.
    pub fn root_children(&self) -> std::collections::btree_map::Iter<'_, Vec<u8>, VocabPrefixTreeNode> {
        self.root.iter_children()
    }

    /// Returns the maximum token ID used to build this tree.
    pub fn max_token_id(&self) -> usize {
        self.max_token_id
    }

    // ----------------------- Legacy (unused) -----------------------
    // Keeping this here for reference; it's no longer used in build().
    // It was replaced by the much faster path-based propagation above.
    #[allow(dead_code)]
    fn compute_reachable_ids_recursive(node: &mut VocabPrefixTreeNode, _max_token_id: usize) -> HashSet<usize> { // max_token_id not used
        let mut current_node_ids_set = HashSet::new();
        current_node_ids_set.insert(node.token_id);
        for child_node in node.children.values_mut() {
            let child_ids_set = Self::compute_reachable_ids_recursive(child_node, _max_token_id);
            current_node_ids_set.extend(child_ids_set);
        }
        let mut final_bitvec = HybridBitset::zeros();
        for token_id_val in &current_node_ids_set {
            final_bitvec.insert(*token_id_val);
        }
        node.reachable_token_ids = final_bitvec;
        current_node_ids_set
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
    use bitvec::prelude::*; // Still needed for macro use perhaps?
    use crate::datastructures::hybrid_bitset::HybridBitset; // Explicitly import HybridBitset
    use std::collections::HashSet;
    use std::iter::FromIterator;


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
        assert!(tree.root.reachable_token_ids.is_empty());
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
        let expected_node_bits = HybridBitset::from_iter(vec![1]);
        assert_eq!(node.reachable_token_ids, expected_node_bits);

        assert_eq!(tree.find_token(&b("hello")), Some(1));
        assert_eq!(tree.find_token(&b("hell")), None);
        assert_eq!(tree.find_token(&b("hello world")), None);
        assert_eq!(tree.find_token(b""), None); // Flag is false

        // Root's reachable IDs should contain only ID 1 (ID 0 is conventional)
        let expected_root_bits = HybridBitset::from_iter(vec![1]);
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
        let expected_node_a_bits = HybridBitset::from_iter(vec![1]);
        assert_eq!(node_a.reachable_token_ids, expected_node_a_bits);

        assert_eq!(tree.find_token(&b("")), Some(99)); // Query for "" returns its ID (flag is true)
        assert_eq!(tree.find_token(&b("a")), Some(1));

        // Root's reachable IDs should contain 1 (from child) and 99 (itself)
        let expected_root_bits = HybridBitset::from_iter(vec![1, 99]);
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
        let expected_node_a_bits = HybridBitset::from_iter(vec![1]);
        assert_eq!(node_a.reachable_token_ids, expected_node_a_bits);

        assert_eq!(tree.find_token(&b("")), Some(0)); // Query for "" returns its ID 0 (flag is true)
        assert_eq!(tree.find_token(&b("a")), Some(1));

        // Root's reachable IDs should contain 1 (from child) and 0 (itself, as it's explicit)
        let expected_root_bits = HybridBitset::from_iter(vec![0, 1]);
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
        let expected_ab_bits = HybridBitset::from_iter(vec![2]);
        assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

        // Node "a" reachable IDs: {1, 2}
        let expected_a_bits = HybridBitset::from_iter(vec![1, 2]);
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
        let expected_abc_bits = HybridBitset::from_iter(vec![3]);
        assert_eq!(node_abc.reachable_token_ids, expected_abc_bits);

        // Node "ab" reachable: {2, 3}
        let expected_ab_bits = HybridBitset::from_iter(vec![2, 3]);
        assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

        // Node "a" reachable: {1, 2, 3}
        let expected_a_bits = HybridBitset::from_iter(vec![1, 2, 3]);
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
        let expected_apple_bits = HybridBitset::from_iter(vec![10]);
        assert_eq!(node_apple.reachable_token_ids, expected_apple_bits);

        assert!(node_app.children.contains_key(&b("ly")));
        let node_apply = node_app.children.get(&b("ly")).unwrap();
        assert_eq!(node_apply.prefix_length, 5); // "apply" length 5
        assert_eq!(node_apply.token_id, 20);
        assert!(node_apply.children.is_empty());
        // Node "apply" reachable: {20}
        let expected_apply_bits = HybridBitset::from_iter(vec![20]);
        assert_eq!(node_apply.reachable_token_ids, expected_apply_bits);

        // Node "app" reachable: {5, 10, 20}
        let expected_app_bits = HybridBitset::from_iter(vec![5, 10, 20]);
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
        assert_eq!(tree.root.children.len(), 2); // "apple" and "apply" are direct children of root
        assert!(tree.root.children.contains_key(&b("apple")));
        assert!(tree.root.children.contains_key(&b("apply")));

        let node_apple = tree.root.children.get(&b("apple")).unwrap();
        assert_eq!(node_apple.prefix_length, 5);
        assert_eq!(node_apple.token_id, 10);
        assert!(node_apple.children.is_empty());
        // Node "apple" reachable: {10}
        let expected_apple_bits = HybridBitset::from_iter(vec![10]);
        assert_eq!(node_apple.reachable_token_ids, expected_apple_bits);

        let node_apply = tree.root.children.get(&b("apply")).unwrap();
        assert_eq!(node_apply.prefix_length, 5);
        assert_eq!(node_apply.token_id, 20);
        assert!(node_apply.children.is_empty());
        // Node "apply" reachable: {20}
        let expected_apply_bits = HybridBitset::from_iter(vec![20]);
        assert_eq!(node_apply.reachable_token_ids, expected_apply_bits);

        assert_eq!(tree.find_token(&b("apple")), Some(10));
        assert_eq!(tree.find_token(&b("apply")), Some(20));
        assert_eq!(tree.find_token(&b("app")), None);
        assert_eq!(tree.find_token(&b("appl")), None);

        // Root reachable: {10, 20}
        let expected_root_bits = HybridBitset::from_iter(vec![10, 20]);
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
        assert_eq!(tree.root.children.len(), 2); // "a" and "b" are direct children
        assert!(tree.root.children.contains_key(&b("a")));
        assert!(tree.root.children.contains_key(&b("b")));

        // Check branch 'a'
        let node_a = tree.root.children.get(&b("a")).unwrap();
        assert_eq!(node_a.token_id, 1);
        assert_eq!(node_a.prefix_length, 1);
        assert_eq!(node_a.children.len(), 1); // "a" is prefix of "ape", "apple", "apply". "ape" is shortest.
                                             // So, "a" -> "pe" (for "ape")
                                             // "ape" -> "le" (for "apple")
                                             // "ape" -> "ly" (for "apply")
                                             // This structure is due to merge_nodes logic.
        assert!(node_a.children.contains_key(&b("pe"))); // Edge from "a" to "ape" node is "pe"

        let node_ape = node_a.children.get(&b("pe")).unwrap();
        assert_eq!(node_ape.token_id, 10);
        assert_eq!(node_ape.prefix_length, 3); // "ape"
        assert_eq!(node_ape.children.len(), 2); // "apple" and "apply" are children of "ape"
        assert!(node_ape.children.contains_key(&b("le"))); // Edge from "ape" to "apple" is "le"
        assert!(node_ape.children.contains_key(&b("ly"))); // Edge from "ape" to "apply" is "ly"

        let node_apple = node_ape.children.get(&b("le")).unwrap();
        assert_eq!(node_apple.token_id, 11);
        assert_eq!(node_apple.prefix_length, 5); // "apple"

        let node_apply = node_ape.children.get(&b("ly")).unwrap();
        assert_eq!(node_apply.token_id, 12);
        assert_eq!(node_apply.prefix_length, 5); // "apply"


        // Node "a" reachable: {1, 10, 11, 12}
        let expected_a_bits = HybridBitset::from_iter(vec![1, 10, 11, 12]);
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
        let expected_b_bits = HybridBitset::from_iter(vec![2, 20]);
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
        let expected_root_bits = HybridBitset::from_iter(vec![1, 2, 10, 11, 12, 20, 99]);
        assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
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
        let expected_ab_bits = HybridBitset::from_iter(vec![2]);
        assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

        // Node "a" reachable: {2, 3} (ID 1 was overwritten)
        let expected_a_bits = HybridBitset::from_iter(vec![2, 3]);
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
            (10, b("ape")), // "a" is prefix of "ape"
            (20, b("banana")), // "b" is prefix of "banana"
        ];
        let tree = VocabPrefixTree::build(&tokens);

        // Iterate root children: "a" and "b"
        let mut root_children_iter = tree.root_children();

        let (edge_a, node_a_ref) = root_children_iter.next().unwrap();
        assert_eq!(edge_a, &b("a"));
        assert_eq!(node_a_ref.token_id, 1);
        assert_eq!(node_a_ref.prefix_length, 1);

        let (edge_b, node_b_ref) = root_children_iter.next().unwrap();
        assert_eq!(edge_b, &b("b"));
        assert_eq!(node_b_ref.token_id, 2);
        assert_eq!(node_b_ref.prefix_length, 1);

        assert!(root_children_iter.next().is_none());

        // Iterate children of node 'a': "pe" (for "ape")
        let mut node_a_children_iter = node_a_ref.iter_children();
        let (edge_pe, node_ape_ref) = node_a_children_iter.next().unwrap();
        assert_eq!(edge_pe, &b("pe"));
        assert_eq!(node_ape_ref.token_id, 10);
        assert_eq!(node_ape_ref.prefix_length, 3); // "ape"
        assert!(node_a_children_iter.next().is_none());

        // Iterate children of node 'b': "anana" (for "banana")
        let mut node_b_children_iter = node_b_ref.iter_children();
        let (edge_anana, node_banana_ref) = node_b_children_iter.next().unwrap();
        assert_eq!(edge_anana, &b("anana"));
        assert_eq!(node_banana_ref.token_id, 20);
        assert_eq!(node_banana_ref.prefix_length, 6); // "banana"
        assert!(node_b_children_iter.next().is_none());
    }

    #[test]
    fn test_find_longest_prefix_token() {
        let tokens = vec![
            (1, b("a")),
            (10, b("ape")),
            (11, b("apple")),
            (12, b("apply")),
            (20, b("banana")),
            (99, b("")), // Empty string token
        ];
        let tree = VocabPrefixTree::build(&tokens);

        // Test case 1: Exact match for "apple"
        assert_eq!(tree.find_longest_prefix_token(b"apple"), Some((11, &b("apple")[..])));

        // Test case 2: Input is longer than any token, "apple" is longest prefix
        assert_eq!(tree.find_longest_prefix_token(b"applepie"), Some((11, &b("apple")[..])));

        // Test case 3: Input is "apply", exact match
        assert_eq!(tree.find_longest_prefix_token(b"apply"), Some((12, &b("apply")[..])));

        // Test case 4: Input is "ape", exact match
        assert_eq!(tree.find_longest_prefix_token(b"ape"), Some((10, &b("ape")[..])));

        // Test case 5: Input is "ap", "a" is the longest prefix token
        assert_eq!(tree.find_longest_prefix_token(b"ap"), Some((1, &b("a")[..])));

        // Test case 6: Input is "application", "apply" is not a prefix, "a" is.
        assert_eq!(tree.find_longest_prefix_token(b"application"), Some((1, &b("a")[..])));


        // Test case 7: Input is "banana", exact match
        assert_eq!(tree.find_longest_prefix_token(&b("banana")), Some((20, &b("banana")[..])));

        // Test case 8: Input is "bananatart", "banana" is longest prefix
        assert_eq!(tree.find_longest_prefix_token(&b("bananatart")), Some((20, &b("banana")[..])));

        // Test case 9: Input is "b", no token starts with "b" other than "banana"
        // Since "" is a token, it should be returned.
        assert_eq!(tree.find_longest_prefix_token(b"b"), Some((99, &b("")[..])));


        // Test case 10: Input is "c", no token starts with "c". "" is a token.
        assert_eq!(tree.find_longest_prefix_token(b"c"), Some((99, &b("")[..])));

        // Test case 11: Input is "", "" is a token.
        assert_eq!(tree.find_longest_prefix_token(b""), Some((99, &b("")[..])));

        // Test case 12: Tree without empty string token
        let tokens_no_empty = vec![
            (1, b("a")),
            (11, b("apple")),
        ];
        let tree_no_empty = VocabPrefixTree::build(&tokens_no_empty);

        assert_eq!(tree_no_empty.find_longest_prefix_token(b"applepie"), Some((11, &b("apple")[..])));
        assert_eq!(tree_no_empty.find_longest_prefix_token(b"ax"), Some((1, &b("a")[..])));
        // Input "b", no token is a prefix, and "" is not a token.
        assert_eq!(tree_no_empty.find_longest_prefix_token(b"b"), None);
        // Input "", "" is not a token.
        assert_eq!(tree_no_empty.find_longest_prefix_token(b""), None);

        // Test case 13: Only empty string token
        let tokens_only_empty = vec![(55, b(""))];
        let tree_only_empty = VocabPrefixTree::build(&tokens_only_empty);
        assert!(tree_only_empty.has_empty_string_token());
        assert_eq!(tree_only_empty.find_longest_prefix_token(b"abc"), Some((55, &b("")[..])));
        assert_eq!(tree_only_empty.find_longest_prefix_token(b""), Some((55, &b("")[..])));

        // Test case 14: Empty tree
        let empty_tokens: Vec<(usize, Vec<u8>)> = vec![];
        let tree_empty = VocabPrefixTree::build(&empty_tokens);
        assert!(!tree_empty.has_empty_string_token());
        assert_eq!(tree_empty.find_longest_prefix_token(b"abc"), None);
        assert_eq!(tree_empty.find_longest_prefix_token(b""), None);

        // Test case 15: Tokens with spaces
        let space_tokens = vec![
            (31, b(" ")),
            (32, b("  ")),
            (34, b("    ")),
        ];
        let tree_spaces = VocabPrefixTree::build(&space_tokens);
        assert!(!tree_spaces.has_empty_string_token());

        // Exact match for one space
        assert_eq!(tree_spaces.find_longest_prefix_token(b" "), Some((31, &b(" ")[..])));
        // Exact match for two spaces
        assert_eq!(tree_spaces.find_longest_prefix_token(b"  "), Some((32, &b("  ")[..])));
        // Input is three spaces, longest prefix is two spaces
        assert_eq!(tree_spaces.find_longest_prefix_token(b"   "), Some((32, &b("  ")[..])));
        // Exact match for four spaces
        assert_eq!(tree_spaces.find_longest_prefix_token(b"    "), Some((34, &b("    ")[..])));
        // Input is five spaces, longest prefix is four spaces
        assert_eq!(tree_spaces.find_longest_prefix_token(b"     "), Some((34, &b("    ")[..])));
        // No match
        assert_eq!(tree_spaces.find_longest_prefix_token(b"a"), None);
        // Empty input, no empty token
        assert_eq!(tree_spaces.find_longest_prefix_token(b""), None);
    }
}
