#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fmt;

use range_set_blaze::RangeSetBlaze;

#[derive(PartialEq)]
pub struct VocabPrefixTreeNode {
    token_id: usize,
    has_token: bool,
    prefix: Box<[u8]>,
    prefix_length: usize,
    children: BTreeMap<Vec<u8>, VocabPrefixTreeNode>,
    reachable_token_ids: RangeSetBlaze<usize>,
}

impl VocabPrefixTreeNode {
    fn new(token_id: usize, prefix: Box<[u8]>, has_token: bool) -> Self {
        let prefix_length = prefix.len();
        Self {
            token_id,
            has_token,
            prefix,
            prefix_length,
            children: BTreeMap::new(),
            reachable_token_ids: RangeSetBlaze::new(),
        }
    }

    pub fn token_id(&self) -> usize {
        self.token_id
    }

    pub fn has_token(&self) -> bool {
        self.has_token
    }

    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    pub fn children(&self) -> &BTreeMap<Vec<u8>, VocabPrefixTreeNode> {
        &self.children
    }

    pub fn iter_children(&self) -> std::collections::btree_map::Iter<'_, Vec<u8>, VocabPrefixTreeNode> {
        self.children.iter()
    }

    pub fn reachable_token_ids(&self) -> &RangeSetBlaze<usize> {
        &self.reachable_token_ids
    }
}

impl fmt::Debug for VocabPrefixTreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn format_bytes(bytes: &[u8]) -> String {
            const MAX_BYTES_DISPLAY: usize = 10;
            let display_str = String::from_utf8_lossy(bytes.get(..MAX_BYTES_DISPLAY).unwrap_or(bytes));
            if bytes.len() > MAX_BYTES_DISPLAY {
                format!("{}...({} bytes)", display_str, bytes.len())
            } else {
                display_str.to_string()
            }
        }

        let mut debug_struct = f.debug_struct("VocabPrefixTreeNode");
        debug_struct.field("token_id", &self.token_id);
        debug_struct.field("has_token", &self.has_token);
        debug_struct.field("prefix_length", &self.prefix_length);
        let reachable_summary = format!("{} items", self.reachable_token_ids.len());
        debug_struct.field("reachable_token_ids", &reachable_summary);
        let children_summary: BTreeMap<String, String> = self
            .iter_children()
            .map(|(k, _)| (format_bytes(k), "<VocabPrefixTreeNode ...>".to_string()))
            .collect();
        debug_struct.field("children", &children_summary);
        debug_struct.finish()
    }
}

#[derive(Debug, PartialEq)]
pub struct VocabPrefixTree {
    pub root: VocabPrefixTreeNode,
    max_token_id: usize,
    has_empty_string_token: bool,
}

impl VocabPrefixTree {
    pub fn new() -> Self {
        Self {
            root: VocabPrefixTreeNode::new(0, Vec::new().into_boxed_slice(), false),
            max_token_id: 0,
            has_empty_string_token: false,
        }
    }

    pub fn build(tokens: &[(usize, Vec<u8>)]) -> Self {
        let mut tree = Self::new();
        tree.max_token_id = tokens.iter().map(|(id, _)| *id).max().unwrap_or(0);

        let mut nonempty: Vec<(usize, Box<[u8]>)> = Vec::with_capacity(tokens.len());
        for (id, bytes) in tokens {
            if bytes.is_empty() {
                tree.root.token_id = *id;
                tree.root.has_token = true;
                tree.has_empty_string_token = true;
            } else {
                nonempty.push((*id, bytes.clone().into_boxed_slice()));
            }
        }

        if nonempty.is_empty() {
            tree.recompute_reachable_ids_via_paths();
            return tree;
        }

        nonempty.sort_by(|a, b| a.1.as_ref().cmp(b.1.as_ref()));

        let mut unique: Vec<(usize, Box<[u8]>)> = Vec::with_capacity(nonempty.len());
        for (id, bytes) in nonempty.into_iter() {
            if let Some((last_id, last_bytes)) = unique.last_mut() {
                if last_bytes.as_ref() == bytes.as_ref() {
                    *last_id = id;
                    continue;
                }
            }
            unique.push((id, bytes));
        }

        #[derive(Debug)]
        struct TmpNode {
            token_id: usize,
            prefix: Box<[u8]>,
            children: Vec<usize>,
        }

        let mut nodes: Vec<TmpNode> = Vec::with_capacity(unique.len() + 1);
        nodes.push(TmpNode {
            token_id: tree.root.token_id,
            prefix: Box::<[u8]>::from(&[][..]),
            children: Vec::new(),
        });

        let mut stack: Vec<usize> = Vec::with_capacity(256);
        stack.push(0);
        let mut prev_bytes: &[u8] = &[];

        for (id, bytes) in unique.into_iter() {
            let mut common = 0usize;
            let maxl = prev_bytes.len().min(bytes.len());
            while common < maxl && prev_bytes[common] == bytes[common] {
                common += 1;
            }

            while let Some(&top_idx) = stack.last() {
                if nodes[top_idx].prefix.len() <= common {
                    break;
                }
                stack.pop();
            }

            while let Some(&top_idx) = stack.last() {
                let top_pref = nodes[top_idx].prefix.as_ref();
                if bytes.starts_with(top_pref) {
                    break;
                }
                stack.pop();
            }

            let parent_idx = *stack.last().unwrap_or(&0);
            let cur_idx = nodes.len();
            nodes.push(TmpNode {
                token_id: id,
                prefix: bytes,
                children: Vec::new(),
            });
            nodes[parent_idx].children.push(cur_idx);
            stack.push(cur_idx);
            prev_bytes = nodes[cur_idx].prefix.as_ref();
        }

        fn finalize(
            idx: usize,
            nodes: &mut [TmpNode],
            root_has_token: bool,
        ) -> VocabPrefixTreeNode {
            let token_id = nodes[idx].token_id;
            let prefix = std::mem::take(&mut nodes[idx].prefix);
            let mut out_node = VocabPrefixTreeNode {
                token_id,
                has_token: idx != 0 || root_has_token,
                prefix,
                prefix_length: 0,
                children: BTreeMap::new(),
                reachable_token_ids: RangeSetBlaze::new(),
            };
            out_node.prefix_length = out_node.prefix.len();
            let parent_prefix_len = out_node.prefix_length;
            let child_indices = std::mem::take(&mut nodes[idx].children);
            for child_idx in child_indices {
                let child_node = finalize(child_idx, nodes, root_has_token);
                let edge_label = child_node.prefix[parent_prefix_len..].to_vec();
                out_node.children.insert(edge_label, child_node);
            }
            out_node
        }

        tree.root = finalize(0, &mut nodes, tree.has_empty_string_token);
        tree.recompute_reachable_ids_via_paths();

        tree
    }

    fn recompute_reachable_ids_via_paths(&mut self) {
        Self::clear_reachable_ids_recursive(&mut self.root);
        let mut ancestor_stack: Vec<*mut VocabPrefixTreeNode> = Vec::new();
        let root_ptr: *mut VocabPrefixTreeNode = &mut self.root;
        unsafe {
            Self::propagate_reachable_ids_dfs(root_ptr, &mut ancestor_stack);
        }
    }

    unsafe fn propagate_reachable_ids_dfs(
        node_ptr: *mut VocabPrefixTreeNode,
        ancestors: &mut Vec<*mut VocabPrefixTreeNode>,
    ) {
        let has_token = unsafe { (*node_ptr).has_token };
        if has_token {
            unsafe {
                (*node_ptr).reachable_token_ids.insert((*node_ptr).token_id);
            }

            let token_id = unsafe { (*node_ptr).token_id };
            for &ancestor_ptr in ancestors.iter() {
                unsafe {
                    (*ancestor_ptr).reachable_token_ids.insert(token_id);
                }
            }
        }

        let mut child_ptrs: Vec<*mut VocabPrefixTreeNode> = Vec::new();
        {
            let current: &mut VocabPrefixTreeNode = unsafe { &mut *node_ptr };
            for child in current.children.values_mut() {
                child_ptrs.push(child as *mut VocabPrefixTreeNode);
            }
        }

        ancestors.push(node_ptr);
        for child_ptr in child_ptrs {
            unsafe {
                Self::propagate_reachable_ids_dfs(child_ptr, ancestors);
            }
        }
        ancestors.pop();
    }

    fn clear_reachable_ids_recursive(node: &mut VocabPrefixTreeNode) {
        node.reachable_token_ids = RangeSetBlaze::new();
        for child in node.children.values_mut() {
            Self::clear_reachable_ids_recursive(child);
        }
    }

    pub fn find_token(&self, bytes: &[u8]) -> Option<usize> {
        if bytes.is_empty() {
            return self.has_empty_string_token.then_some(self.root.token_id);
        }

        let mut current_node = &self.root;
        let mut remaining_bytes = bytes;

        loop {
            let mut found_match = false;
            let first = remaining_bytes[0];
            let lower_key = vec![first];
            for (edge_label, child_node) in current_node.children.range(lower_key..) {
                if edge_label.first().copied() != Some(first) {
                    break;
                }
                if remaining_bytes.starts_with(edge_label) {
                    remaining_bytes = &remaining_bytes[edge_label.len()..];
                    current_node = child_node;
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                return None;
            }

            if remaining_bytes.is_empty() {
                return Some(current_node.token_id);
            }
        }
    }

    pub fn find_longest_prefix_token<'s>(&'s self, bytes: &[u8]) -> Option<(usize, &'s [u8])> {
        let mut longest_match_info: Option<(usize, &'s [u8])> = None;
        let mut current_node: &'s VocabPrefixTreeNode = &self.root;

        if self.has_empty_string_token {
            longest_match_info = Some((self.root.token_id(), self.root.prefix()));
        }

        if bytes.is_empty() {
            return longest_match_info;
        }

        let mut remaining_bytes = bytes;
        loop {
            let mut found_match_in_children = false;
            let first = remaining_bytes[0];
            let lower_key = vec![first];
            for (edge_label, child_node) in current_node.children.range(lower_key..) {
                if edge_label.first().copied() != Some(first) {
                    break;
                }
                if remaining_bytes.starts_with(edge_label) {
                    current_node = child_node;
                    remaining_bytes = &remaining_bytes[edge_label.len()..];
                    longest_match_info = Some((current_node.token_id(), current_node.prefix()));
                    found_match_in_children = true;
                    break;
                }
            }

            if !found_match_in_children || remaining_bytes.is_empty() {
                break;
            }
        }
        longest_match_info
    }

    pub fn has_empty_string_token(&self) -> bool {
        self.has_empty_string_token
    }

    pub fn root_children(&self) -> std::collections::btree_map::Iter<'_, Vec<u8>, VocabPrefixTreeNode> {
        self.root.iter_children()
    }

    pub fn max_token_id(&self) -> usize {
        self.max_token_id
    }
}

impl Default for VocabPrefixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Eq for VocabPrefixTreeNode {}

impl PartialOrd for VocabPrefixTreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VocabPrefixTreeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prefix_length
            .cmp(&other.prefix_length)
            .then_with(|| self.token_id.cmp(&other.token_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_find_prefix_tokens() {
        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"abc".to_vec()),
            (3, b"b".to_vec()),
        ]);

        assert_eq!(tree.find_token(b"a"), Some(0));
        assert_eq!(tree.find_token(b"ab"), Some(1));
        assert_eq!(tree.find_token(b"abc"), Some(2));
        assert_eq!(tree.find_token(b"b"), Some(3));
        assert_eq!(tree.find_token(b"ac"), None);
        assert_eq!(tree.find_longest_prefix_token(b"abcd").map(|(id, _)| id), Some(2));
    }
}
