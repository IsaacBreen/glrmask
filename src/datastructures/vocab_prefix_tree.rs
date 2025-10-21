use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fmt;

use bitvec::prelude::*;
use range_set_blaze::RangeSetBlaze;
// Keep for macros or other uses if needed
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::json_serialization::{JSONConvertible, JSONNode};
// Added
use std::collections::BTreeMap as StdMap;
// Added for derive macro pattern


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
    reachable_token_ids: RangeSetBlaze<usize>,
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
            reachable_token_ids: RangeSetBlaze::new(),
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
    pub fn reachable_token_ids(&self) -> &RangeSetBlaze<usize> {
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
            "{} items",
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
        node.reachable_token_ids = RangeSetBlaze::new();
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

    /// Finds the longest token in the tree that is a prefix of the given `bytes`.
    ///
    /// Returns `Some((token_id, matched_prefix_bytes))` if a match is found.
    /// The `matched_prefix_bytes` is a slice of the token's full byte sequence stored in the tree.
    ///
    /// If the input `bytes` is empty:
    ///  - If the empty string `""` is a token in the tree, it returns `Some((empty_token_id, &[]))`.
    ///  - Otherwise, it returns `None`.
    ///
    /// If the input `bytes` is not empty:
    ///  - It searches for the longest token that is a prefix of `bytes`.
    ///  - If the empty string `""` is a token and no non-empty token prefix is found,
    ///    it will return `Some((empty_token_id, &[]))`.
    ///  - If no token (including potentially the empty string) is a prefix, it returns `None`.
    pub fn find_longest_prefix_token<'s>(&'s self, bytes: &[u8]) -> Option<(usize, &'s [u8])> {
        let mut longest_match_info: Option<(usize, &'s [u8])> = None;
        let mut current_node: &'s VocabPrefixTreeNode = &self.root;

        // Handle empty string token possibility upfront.
        // If it exists, it's a candidate for the longest prefix.
        if self.has_empty_string_token {
            longest_match_info = Some((self.root.token_id(), self.root.prefix())); // prefix is &[]
        }

        // If the input `bytes` itself is empty, the only possible match is the empty string token (if it exists).
        if bytes.is_empty() {
            return longest_match_info;
        }

        let mut remaining_bytes = bytes;

        // Traverse the tree along the path defined by the input `bytes`.
        // Every node landed on is a token, and thus a candidate for the longest prefix match.
        loop {
            let mut found_match_in_children = false;
            for (edge_label, child_node) in current_node.children() {
                if remaining_bytes.starts_with(edge_label) {
                    // Descend to the child node.
                    current_node = child_node;
                    remaining_bytes = &remaining_bytes[edge_label.len()..];

                    // This child_node represents a token. Its full prefix is `current_node.prefix()`.
                    // This token is a prefix of the original `bytes` input.
                    // Update longest_match_info as this is a longer or equally long (but later found) prefix.
                    longest_match_info = Some((current_node.token_id(), current_node.prefix()));

                    found_match_in_children = true;
                    break; // Continue traversal from the new current_node.
                }
            }

            if !found_match_in_children {
                // No child edge matches the start of the remaining_bytes.
                // Cannot extend the prefix further. The current longest_match_info is the result.
                break;
            }

            if remaining_bytes.is_empty() {
                // All input bytes have been consumed along tree edges.
                // The token corresponding to the current_node is the longest possible match.
                // (This was already updated when we landed on current_node).
                break;
            }
        }
        longest_match_info
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
        let mut final_bitvec = RangeSetBlaze::new();
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

#[cfg(test)]
#[path = "vocab_prefix_tree_tests.rs"]
mod tests;
