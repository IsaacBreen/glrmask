use std::fmt;

use range_set_blaze::RangeSetBlaze;

#[derive(PartialEq)]
pub struct VocabPrefixTreeNode {
    token_id: usize,
    has_token: bool,
    prefix: Box<[u8]>,

    // Packed child storage.
    // Edge labels are implicit:
    // child_edge_label = &child.prefix[self.prefix.len()..]
    children: Box<[VocabPrefixTreeNode]>,

    reachable_token_ids: RangeSetBlaze<usize>,
    subtree_bytes: [u64; 4],
}

#[derive(Clone)]
pub struct VocabPrefixTreeChildIter<'a> {
    parent_prefix_len: usize,
    inner: std::slice::Iter<'a, VocabPrefixTreeNode>,
}

impl<'a> Iterator for VocabPrefixTreeChildIter<'a> {
    type Item = (&'a [u8], &'a VocabPrefixTreeNode);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let child = self.inner.next()?;
        Some((&child.prefix[self.parent_prefix_len..], child))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for VocabPrefixTreeChildIter<'_> {}

impl VocabPrefixTreeNode {
    #[inline]
    fn new(token_id: usize, prefix: Box<[u8]>, has_token: bool) -> Self {
        Self {
            token_id,
            has_token,
            prefix,
            children: Box::new([]),
            reachable_token_ids: RangeSetBlaze::new(),
            subtree_bytes: [0u64; 4],
        }
    }

    #[inline]
    pub fn token_id(&self) -> usize {
        self.token_id
    }

    #[inline]
    pub fn has_token(&self) -> bool {
        self.has_token
    }

    #[inline]
    pub fn prefix_length(&self) -> usize {
        self.prefix.len()
    }

    #[inline]
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    #[inline]
    pub fn children(&self) -> &[VocabPrefixTreeNode] {
        &self.children
    }

    #[inline]
    pub fn iter_children(&self) -> VocabPrefixTreeChildIter<'_> {
        VocabPrefixTreeChildIter {
            parent_prefix_len: self.prefix.len(),
            inner: self.children.iter(),
        }
    }

    #[inline]
    pub fn reachable_token_ids(&self) -> &RangeSetBlaze<usize> {
        &self.reachable_token_ids
    }

    #[inline]
    pub fn subtree_bytes(&self) -> &[u64; 4] {
        &self.subtree_bytes
    }

    #[inline]
    fn child_edge_label<'a>(&'a self, child: &'a VocabPrefixTreeNode) -> &'a [u8] {
        &child.prefix[self.prefix.len()..]
    }

    #[inline]
    fn child_key_byte(&self, child: &VocabPrefixTreeNode) -> u8 {
        child.prefix[self.prefix.len()]
    }

    #[inline]
    fn find_child(&self, next_byte: u8) -> Option<&VocabPrefixTreeNode> {
        let idx = self
            .children
            .binary_search_by_key(&next_byte, |child| self.child_key_byte(child))
            .ok()?;
        Some(&self.children[idx])
    }
}

impl fmt::Debug for VocabPrefixTreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn format_bytes(bytes: &[u8]) -> String {
            const MAX_BYTES_DISPLAY: usize = 10;
            let display_str =
                String::from_utf8_lossy(bytes.get(..MAX_BYTES_DISPLAY).unwrap_or(bytes));
            if bytes.len() > MAX_BYTES_DISPLAY {
                format!("{}...({} bytes)", display_str, bytes.len())
            } else {
                display_str.to_string()
            }
        }

        let children_summary: Vec<String> = self
            .iter_children()
            .map(|(edge, _)| format_bytes(edge))
            .collect();

        let mut debug_struct = f.debug_struct("VocabPrefixTreeNode");
        debug_struct.field("token_id", &self.token_id);
        debug_struct.field("has_token", &self.has_token);
        debug_struct.field("prefix_length", &self.prefix.len());
        debug_struct.field("prefix", &format_bytes(&self.prefix));
        debug_struct.field(
            "reachable_token_ids",
            &format!("{} items", self.reachable_token_ids.len()),
        );
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
    #[inline]
    pub fn new() -> Self {
        Self {
            root: VocabPrefixTreeNode::new(0, Vec::new().into_boxed_slice(), false),
            max_token_id: 0,
            has_empty_string_token: false,
        }
    }

    pub fn build(tokens: &[(usize, Vec<u8>)]) -> Self {
        Self::build_owned(tokens.iter().map(|(id, bytes)| (*id, bytes.clone())).collect())
    }

    pub fn build_owned(mut tokens: Vec<(usize, Vec<u8>)>) -> Self {
        let mut tree = Self::new();
        tree.max_token_id = tokens.iter().map(|(id, _)| *id).max().unwrap_or(0);

        if tokens.is_empty() {
            return tree;
        }

        // Fast sort for large inputs.
        tokens.sort_unstable_by(|a, b| a.1.as_slice().cmp(b.1.as_slice()));

        // In-place dedup by byte string.
        let mut write = 0usize;
        for read in 0..tokens.len() {
            if write > 0 && tokens[write - 1].1 == tokens[read].1 {
                tokens[write - 1].0 = tokens[read].0;
            } else {
                if write != read {
                    tokens.swap(write, read);
                }
                write += 1;
            }
        }
        tokens.truncate(write);

        // Handle empty token if present.
        let mut start = 0usize;
        if !tokens.is_empty() && tokens[0].1.is_empty() {
            tree.root.token_id = tokens[0].0;
            tree.root.has_token = true;
            tree.has_empty_string_token = true;
            tree.root.reachable_token_ids.insert(tokens[0].0);
            start = 1;
        }

        if start == tokens.len() {
            return tree;
        }

        let entries = &tokens[start..];
        let mut children = Vec::new();
        let mut root_subtree_bytes = [0u64; 4];

        let mut i = 0usize;
        while i < entries.len() {
            let byte0 = entries[i].1[0];
            let group_start = i;
            i += 1;
            while i < entries.len() && entries[i].1[0] == byte0 {
                i += 1;
            }

            let child = Self::build_subtree(&entries[group_start..i], 0);

            for token_id in child.reachable_token_ids.iter() {
                tree.root.reachable_token_ids.insert(token_id);
            }

            let edge = &child.prefix[..];
            for &b in edge {
                root_subtree_bytes[b as usize >> 6] |= 1u64 << (b & 63);
            }
            for j in 0..4 {
                root_subtree_bytes[j] |= child.subtree_bytes[j];
            }

            children.push(child);
        }

        tree.root.children = children.into_boxed_slice();
        tree.root.subtree_bytes = root_subtree_bytes;

        tree
    }

    #[inline]
    fn lcp_len(a: &[u8], b: &[u8], from: usize) -> usize {
        let max_len = a.len().min(b.len());
        let mut i = from;
        while i < max_len && a[i] == b[i] {
            i += 1;
        }
        i
    }

    fn build_subtree(entries: &[(usize, Vec<u8>)], parent_prefix_len: usize) -> VocabPrefixTreeNode {
        debug_assert!(!entries.is_empty());

        let first = entries.first().unwrap().1.as_slice();
        let last = entries.last().unwrap().1.as_slice();

        // For lexicographically sorted entries, the common prefix of first and last
        // is the common prefix of the entire group.
        let prefix_len = Self::lcp_len(first, last, parent_prefix_len);
        let has_token = entries[0].1.len() == prefix_len;
        let token_id = if has_token { entries[0].0 } else { 0 };
        let prefix = first[..prefix_len].to_vec().into_boxed_slice();

        let child_entries = if has_token { &entries[1..] } else { entries };
        let mut children = Vec::new();

        let mut i = 0usize;
        while i < child_entries.len() {
            let next_byte = child_entries[i].1[prefix_len];
            let group_start = i;
            i += 1;
            while i < child_entries.len() && child_entries[i].1[prefix_len] == next_byte {
                i += 1;
            }
            children.push(Self::build_subtree(
                &child_entries[group_start..i],
                prefix_len,
            ));
        }

        let mut reachable_token_ids = RangeSetBlaze::new();
        if has_token {
            reachable_token_ids.insert(token_id);
        }

        let mut subtree_bytes = [0u64; 4];
        for child in &children {
            let edge = &child.prefix[prefix_len..];
            for &b in edge {
                subtree_bytes[b as usize >> 6] |= 1u64 << (b & 63);
            }
            for j in 0..4 {
                subtree_bytes[j] |= child.subtree_bytes[j];
            }
            for token_id in child.reachable_token_ids.iter() {
                reachable_token_ids.insert(token_id);
            }
        }

        VocabPrefixTreeNode {
            token_id,
            has_token,
            prefix,
            children: children.into_boxed_slice(),
            reachable_token_ids,
            subtree_bytes,
        }
    }

    #[inline]
    pub fn find_token(&self, bytes: &[u8]) -> Option<usize> {
        if bytes.is_empty() {
            return self.has_empty_string_token.then_some(self.root.token_id);
        }

        let mut current = &self.root;
        let mut remaining = bytes;

        loop {
            let child = current.find_child(remaining[0])?;
            let edge = current.child_edge_label(child);

            if !remaining.starts_with(edge) {
                return None;
            }

            remaining = &remaining[edge.len()..];
            current = child;

            if remaining.is_empty() {
                return current.has_token.then_some(current.token_id);
            }
        }
    }

    #[inline]
    pub fn find_longest_prefix_token<'s>(&'s self, bytes: &[u8]) -> Option<(usize, &'s [u8])> {
        let mut best = None;

        let mut current = &self.root;
        let mut remaining = bytes;

        if self.has_empty_string_token {
            best = Some((self.root.token_id, self.root.prefix()));
        }

        while !remaining.is_empty() {
            let Some(child) = current.find_child(remaining[0]) else {
                break;
            };

            let edge = current.child_edge_label(child);
            if !remaining.starts_with(edge) {
                break;
            }

            remaining = &remaining[edge.len()..];
            current = child;

            if current.has_token {
                best = Some((current.token_id, current.prefix()));
            }
        }

        best
    }

    #[inline]
    pub fn has_empty_string_token(&self) -> bool {
        self.has_empty_string_token
    }

    #[inline]
    pub fn root_children(&self) -> VocabPrefixTreeChildIter<'_> {
        self.root.iter_children()
    }

    #[inline]
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

    #[test]
    fn internal_branch_nodes_are_not_tokens() {
        let tree = VocabPrefixTree::build(&[
            (10, b"abcd".to_vec()),
            (11, b"abef".to_vec()),
        ]);

        assert_eq!(tree.find_token(b"ab"), None);
        assert_eq!(tree.find_token(b"abcd"), Some(10));
        assert_eq!(tree.find_token(b"abef"), Some(11));
        assert_eq!(tree.find_longest_prefix_token(b"abzz"), None);
    }

    #[test]
    fn empty_string_token_works() {
        let tree = VocabPrefixTree::build(&[
            (7, b"".to_vec()),
            (8, b"a".to_vec()),
            (9, b"abc".to_vec()),
        ]);

        assert!(tree.has_empty_string_token());
        assert_eq!(tree.find_token(b""), Some(7));
        assert_eq!(tree.find_longest_prefix_token(b"zzz").map(|(id, _)| id), Some(7));
        assert_eq!(tree.find_longest_prefix_token(b"abcd").map(|(id, _)| id), Some(9));
    }

    #[test]
    fn duplicate_tokens_are_deduped() {
        let tree = VocabPrefixTree::build(&[
            (1, b"abc".to_vec()),
            (2, b"abc".to_vec()),
            (3, b"abcd".to_vec()),
        ]);

        assert!(tree.find_token(b"abc").is_some());
        assert_eq!(tree.find_token(b"abcd"), Some(3));
    }

    #[test]
    fn root_stays_empty_prefix() {
        let tree = VocabPrefixTree::build(&[
            (1, b"abc".to_vec()),
            (2, b"abd".to_vec()),
        ]);

        assert_eq!(tree.root.prefix(), b"");
        assert_eq!(tree.root.prefix_length(), 0);
    }

    #[test]
    fn iter_children_returns_edge_and_child() {
        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
        ]);

        let root_children: Vec<(&[u8], usize)> = tree
            .root
            .iter_children()
            .map(|(edge, child)| (edge, child.token_id()))
            .collect();

        assert_eq!(root_children.len(), 2);
        assert_eq!(root_children[0].0, b"a");
        assert_eq!(root_children[1].0, b"b");
    }
}