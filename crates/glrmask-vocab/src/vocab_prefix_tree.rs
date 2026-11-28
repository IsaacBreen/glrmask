use std::fmt;
use std::sync::{Arc, OnceLock};

use range_set_blaze::RangeSetBlaze;

/// A radix node is an interval of lexically ordered spellings, with its
/// common prefix represented by a length into the first spelling. Child IDs
/// occupy a contiguous slice of the tree's child index. No prefixes, token
/// membership lists or separately allocated child nodes are duplicated.
#[derive(Debug, PartialEq, Eq, Default)]
struct RadixNode {
    first: u32,
    end: u32,
    prefix_len: u32,
    child_start: u32,
    child_end: u32,
    subtree_bytes: [u64; 4],
}

#[derive(Clone, Copy)]
pub struct VocabPrefixTreeNode<'a> {
    tree: &'a VocabPrefixTree,
    index: u32,
}

#[derive(Clone)]
pub struct VocabPrefixTreeChildIter<'a> {
    parent_prefix_len: usize,
    tree: &'a VocabPrefixTree,
    inner: std::slice::Iter<'a, u32>,
}

impl<'a> Iterator for VocabPrefixTreeChildIter<'a> {
    type Item = (&'a [u8], VocabPrefixTreeNode<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let child = self.tree.node(*self.inner.next()?);
        Some((&child.prefix()[self.parent_prefix_len..], child))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl<'a> DoubleEndedIterator for VocabPrefixTreeChildIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let child = self.tree.node(*self.inner.next_back()?);
        Some((&child.prefix()[self.parent_prefix_len..], child))
    }
}

impl ExactSizeIterator for VocabPrefixTreeChildIter<'_> {}

impl<'a> VocabPrefixTreeNode<'a> {
    #[inline]
    fn data(self) -> &'a RadixNode { &self.tree.nodes[self.index as usize] }

    #[inline]
    pub fn index(self) -> u32 { self.index }

    /// Stable only for the lifetime of the owning immutable tree.
    #[inline]
    pub fn cache_key(self) -> usize { self.data() as *const RadixNode as usize }

    #[inline]
    pub fn token_id(self) -> usize {
        if self.has_token() { self.tree.id_at(self.data().first as usize) } else { 0 }
    }

    #[inline]
    pub fn has_token(self) -> bool {
        let n = self.data();
        n.first < n.end && self.tree.tokens[n.first as usize].len() == n.prefix_len as usize
    }

    #[inline]
    pub fn prefix_length(self) -> usize { self.data().prefix_len as usize }

    #[inline]
    pub fn prefix(self) -> &'a [u8] {
        let n = self.data();
        if n.prefix_len == 0 { &[] }
        else { &self.tree.tokens[n.first as usize][..n.prefix_len as usize] }
    }

    #[inline]
    pub fn children(self) -> &'a [u32] {
        let n = self.data();
        &self.tree.children[n.child_start as usize..n.child_end as usize]
    }

    #[inline]
    pub fn iter_children(self) -> VocabPrefixTreeChildIter<'a> {
        VocabPrefixTreeChildIter {
            parent_prefix_len: self.prefix_length(), tree: self.tree, inner: self.children().iter(),
        }
    }

    /// Half-open interval in the lexical spelling coordinate.
    #[inline]
    pub fn lexical_interval(self) -> (u32, u32) {
        let n = self.data();
        (n.first, n.end)
    }

    /// Only valid when token IDs themselves are the lexical coordinate.
    #[inline]
    pub fn ordered_interval(self) -> Option<(u32, u32)> {
        self.tree.ids.is_none().then(|| self.lexical_interval())
    }

    #[inline]
    pub fn reachable_token_ids(self) -> &'a RangeSetBlaze<usize> {
        &self.tree.reachable_sets()[self.index as usize]
    }

    #[inline]
    pub fn subtree_bytes(self) -> &'a [u64; 4] { &self.data().subtree_bytes }

    #[inline]
    fn find_child(self, next_byte: u8) -> Option<Self> {
        let prefix_len = self.prefix_length();
        let idx = self.children().binary_search_by_key(&next_byte, |&id|
            self.tree.node(id).prefix()[prefix_len]).ok()?;
        Some(self.tree.node(self.children()[idx]))
    }

    fn next_matching_child<'b>(self, remaining: &'b [u8]) -> Option<(Self, &'b [u8])> {
        let child = self.find_child(*remaining.first()?)?;
        let edge = &child.prefix()[self.prefix_length()..];
        remaining.strip_prefix(edge).map(|suffix| (child, suffix))
    }

    fn reachable_count(self) -> Option<usize> {
        if let Some((start, end)) = self.ordered_interval() { Some((end - start) as usize) }
        else { usize::try_from(self.reachable_token_ids().len()).ok() }
    }
}

impl fmt::Debug for VocabPrefixTreeNode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabPrefixTreeNode")
            .field("token_id", &self.has_token().then(|| self.token_id()))
            .field("prefix_length", &self.prefix_length())
            .field("prefix", &self.prefix().get(..10).unwrap_or(self.prefix()))
            .field("lexical_interval", &self.lexical_interval())
            .field("children", &self.children()).finish()
    }
}

pub struct VocabPrefixTree {
    tokens: Arc<Vec<Arc<[u8]>>>,
    /// Omitted for the common lexical-ID coordinate. A general terminal
    /// quotient may have arbitrary IDs and hence noncontiguous reachable sets.
    ids: Option<Box<[usize]>>,
    nodes: Box<[RadixNode]>,
    children: Box<[u32]>,
    max_token_id: usize,
    reachable: OnceLock<Box<[RangeSetBlaze<usize>]>>,
}

impl fmt::Debug for VocabPrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabPrefixTree").field("tokens", &self.tokens.len())
            .field("nodes", &self.nodes.len()).field("root", &self.root()).finish()
    }
}

impl PartialEq for VocabPrefixTree {
    fn eq(&self, other: &Self) -> bool {
        self.tokens == other.tokens && self.ids == other.ids
            && self.nodes == other.nodes && self.children == other.children
    }
}

impl VocabPrefixTree {
    pub fn new() -> Self { Self::from_ordered_tokens(Arc::new(Vec::new())) }

    pub fn root(&self) -> VocabPrefixTreeNode<'_> { self.node(0) }

    pub fn node(&self, index: u32) -> VocabPrefixTreeNode<'_> {
        VocabPrefixTreeNode { tree: self, index }
    }

    pub fn len(&self) -> usize { self.nodes.len() }

    #[inline]
    fn id_at(&self, ordinal: usize) -> usize {
        self.ids.as_ref().map_or(ordinal, |ids| ids[ordinal])
    }

    fn reachable_sets(&self) -> &[RangeSetBlaze<usize>] {
        self.reachable.get_or_init(|| {
            if self.ids.is_none() {
                return self.nodes.iter().map(|n| {
                    if n.first == n.end { RangeSetBlaze::new() }
                    else { RangeSetBlaze::from_iter([n.first as usize..=n.end as usize - 1]) }
                }).collect::<Vec<_>>().into_boxed_slice();
            }
            let mut sets = vec![RangeSetBlaze::new(); self.nodes.len()];
            // Parents precede children; accumulate general ID sets bottom-up.
            for index in (0..self.nodes.len()).rev() {
                let node = self.node(index as u32);
                let mut set = RangeSetBlaze::new();
                if node.has_token() { set.insert(node.token_id()); }
                for &child in node.children() { set |= &sets[child as usize]; }
                sets[index] = set;
            }
            sets.into_boxed_slice()
        })
    }

    pub fn build(tokens: &[(usize, Vec<u8>)]) -> Self { Self::build_owned(tokens.to_vec()) }

    pub fn build_owned(mut tokens: Vec<(usize, Vec<u8>)>) -> Self {
        tokens.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        let mut write = 0usize;
        for read in 0..tokens.len() {
            if write > 0 && tokens[write - 1].1 == tokens[read].1 {
                tokens[write - 1].0 = tokens[read].0;
                continue;
            }
            if write != read { tokens.swap(write, read); }
            write += 1;
        }
        tokens.truncate(write);
        let (ids, words): (Vec<_>, Vec<Arc<[u8]>>) = tokens.into_iter()
            .map(|(id, bytes)| (id, Arc::from(bytes))).unzip();
        Self::from_parts(Arc::new(words), Some(ids.into_boxed_slice()))
    }

    /// Build from lexically sorted, distinct spellings.
    pub fn build_presorted(tokens: &[(usize, &[u8])]) -> Self {
        let (ids, words): (Vec<_>, Vec<Arc<[u8]>>) = tokens.iter()
            .map(|&(id, bytes)| (id, Arc::from(bytes))).unzip();
        Self::from_parts(Arc::new(words), Some(ids.into_boxed_slice()))
    }

    /// Reuse the vocabulary's immutable, distinct, byte-sorted spellings.
    pub fn from_ordered_tokens(tokens: Arc<Vec<Arc<[u8]>>>) -> Self {
        Self::from_parts(tokens, None)
    }

    fn from_parts(tokens: Arc<Vec<Arc<[u8]>>>, ids: Option<Box<[usize]>>) -> Self {
        debug_assert!(tokens.windows(2).all(|pair| pair[0] < pair[1]));
        let ids = ids.filter(|ids| ids.iter().copied().ne(0..tokens.len()));
        let max_token_id = ids.as_ref().map_or(tokens.len().saturating_sub(1),
            |ids| ids.iter().copied().max().unwrap_or(0));
        fn add_node(tokens: &[Arc<[u8]>], first: usize, end: usize, prefix_len: usize,
            nodes: &mut Vec<RadixNode>, children: &mut Vec<u32>) -> u32
        {
            let index = u32::try_from(nodes.len()).expect("radix node index");
            nodes.push(RadixNode::default());
            let mut cursor = first;
            if cursor < end && tokens[cursor].len() == prefix_len { cursor += 1; }
            let mut local_children = Vec::new();
            let mut subtree_bytes = [0; 4];
            while cursor < end {
                let start = cursor;
                let byte = tokens[cursor][prefix_len];
                cursor += 1;
                while cursor < end && tokens[cursor][prefix_len] == byte { cursor += 1; }
                let a = &tokens[start];
                let b = &tokens[cursor - 1];
                let depth = prefix_len + a[prefix_len..].iter().zip(&b[prefix_len..])
                    .take_while(|(a,b)| a == b).count();
                let child = add_node(tokens, start, cursor, depth, nodes, children);
                for &byte in &a[prefix_len..depth] {
                    subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
                }
                for (dst, src) in subtree_bytes.iter_mut().zip(nodes[child as usize].subtree_bytes) {
                    *dst |= src;
                }
                local_children.push(child);
            }
            let child_start = u32::try_from(children.len()).expect("radix child index");
            children.extend(local_children);
            nodes[index as usize] = RadixNode {
                first: u32::try_from(first).expect("radix spelling index"),
                end: u32::try_from(end).expect("radix spelling index"),
                prefix_len: u32::try_from(prefix_len).expect("radix prefix length"),
                child_start, child_end: u32::try_from(children.len()).expect("radix child index"),
                subtree_bytes,
            };
            index
        }
        let mut nodes = Vec::new();
        let mut children = Vec::new();
        add_node(&tokens, 0, tokens.len(), 0, &mut nodes, &mut children);
        let tree = Self { tokens, ids, nodes: nodes.into_boxed_slice(),
            children: children.into_boxed_slice(), max_token_id, reachable: OnceLock::new() };
        // General ID trees need exact distinct-ID cardinalities for bounded
        // proper-prefix queries, so do that work during preparation.
        if tree.ids.is_some() { tree.reachable_sets(); }
        tree
    }

    /// Return exact IDs whose positive proper prefix contains a selected byte.
    /// Charge all examined nodes and emitted IDs before producing an answer.
    pub fn token_ids_with_proper_prefix_byte(
        &self, bytes: &[u64; 4], mut work_limit: usize,
    ) -> Option<Vec<usize>> {
        if bytes.iter().all(|&word| word == 0) { return Some(Vec::new()); }
        let intersects = |observed: &[u64; 4]| observed.iter().zip(bytes)
            .any(|(&a, &b)| a & b != 0);
        let contains = |byte: u8| bytes[byte as usize >> 6] & (1u64 << (byte & 63)) != 0;
        let mut subtrees = Vec::new();
        let mut token_count = 0usize;
        let mut pending = vec![self.root()];
        while let Some(node) = pending.pop() {
            work_limit = work_limit.checked_sub(1)?;
            if !intersects(node.subtree_bytes()) { continue; }
            for (edge, child) in node.iter_children() {
                work_limit = work_limit.checked_sub(1)?;
                if let Some(offset) = edge.iter().position(|&byte| contains(byte)) {
                    let exclude_endpoint = offset + 1 == edge.len() && child.has_token();
                    let count = child.reachable_count()?.checked_sub(usize::from(exclude_endpoint))?;
                    work_limit = work_limit.checked_sub(count)?;
                    token_count = token_count.checked_add(count)?;
                    if count != 0 { subtrees.push((child, exclude_endpoint)); }
                } else if intersects(child.subtree_bytes()) {
                    pending.push(child);
                }
            }
        }
        let mut ids = Vec::with_capacity(token_count);
        for (child, exclude_endpoint) in subtrees {
            if let Some((first, end)) = child.ordered_interval() {
                ids.extend(first as usize + usize::from(exclude_endpoint)..end as usize);
            } else {
                ids.extend(child.reachable_token_ids().iter().filter(|&id|
                    !exclude_endpoint || id != child.token_id()));
            }
        }
        debug_assert_eq!(ids.len(), token_count);
        Some(ids)
    }

    pub fn find_token(&self, bytes: &[u8]) -> Option<usize> {
        let mut current = self.root();
        let mut remaining = bytes;
        while !remaining.is_empty() {
            (current, remaining) = current.next_matching_child(remaining)?;
        }
        current.has_token().then(|| current.token_id())
    }

    pub fn find_longest_prefix_token<'a>(&'a self, bytes: &[u8]) -> Option<(usize, &'a [u8])> {
        let mut current = self.root();
        let mut remaining = bytes;
        let mut best = current.has_token().then(|| (current.token_id(), current.prefix()));
        while let Some((child, suffix)) = current.next_matching_child(remaining) {
            current = child;
            remaining = suffix;
            if current.has_token() { best = Some((current.token_id(), current.prefix())); }
        }
        best
    }

    pub fn has_empty_string_token(&self) -> bool { self.root().has_token() }

    pub fn root_children(&self) -> VocabPrefixTreeChildIter<'_> { self.root().iter_children() }

    pub fn max_token_id(&self) -> usize { self.max_token_id }
}

impl Default for VocabPrefixTree {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod proper_prefix_byte_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn mask(bytes: impl IntoIterator<Item=u8>) -> [u64; 4] {
        let mut mask = [0;4];
        for byte in bytes { mask[byte as usize >> 6] |= 1u64 << (byte & 63); }
        mask
    }

    fn reference(entries: &[(usize,Vec<u8>)], query: &[u64;4]) -> Vec<usize> {
        let mut ids = entries.iter().filter_map(|(id, bytes)| {
            bytes.get(..bytes.len().saturating_sub(1)).unwrap().iter()
                .any(|&byte| query[byte as usize >> 6] & (1u64 << (byte & 63)) != 0)
                .then_some(*id)
        }).collect::<Vec<_>>();
        ids.sort_unstable(); ids
    }

    #[test]
    fn lexical_radix_intervals_match_literal_prefixes() {
        let words = [b"".as_slice(), b"a", b"abc", b"abd", b"b", b"zz"];
        let tree = VocabPrefixTree::from_ordered_tokens(Arc::new(
            words.iter().map(|word| Arc::<[u8]>::from(*word)).collect()
        ));
        let mut pending = vec![tree.root()];
        while let Some(node) = pending.pop() {
            let (first, end) = node.ordered_interval().unwrap();
            let expected = words.iter().enumerate()
                .filter_map(|(id, word)| word.starts_with(node.prefix()).then_some(id))
                .collect::<Vec<_>>();
            assert_eq!((first as usize..end as usize).collect::<Vec<_>>(), expected);
            assert_eq!(tree.find_token(node.prefix()), node.has_token().then(|| node.token_id()));
            pending.extend(node.iter_children().map(|(_, child)| child));
        }
        assert_eq!(tree.find_longest_prefix_token(b"abcx"), Some((2, b"abc".as_slice())));
        assert_eq!(tree.find_token(b"ab"), None);
        assert!(tree.reachable.get().is_none(), "interval queries need no materialized ID sets");
    }

    #[test]
    fn proper_prefix_byte_query_respects_endpoints_compressed_edges_and_budget() {
        let entries = [b"".as_slice(),b"b",b"ab",b"abc",b"abcd",b"abef",b"bc",
            b"zzzzzzzzzzb",b"zzzzzzzzzzbx",b"\xff",b"\xff\x00",b"x\xff",b"x\xffy"]
            .into_iter().enumerate().map(|(i,b)| (i*19+7,b.to_vec())).collect::<Vec<_>>();
        let tree = VocabPrefixTree::build_owned(entries.clone());
        for query in [mask(*b"a"),mask(*b"b"),mask(*b"c"),mask([255]),mask([0]),mask([255,b'b']),[u64::MAX;4],[0;4]] {
            let expected=reference(&entries,&query);
            let mut actual=tree.token_ids_with_proper_prefix_byte(&query,usize::MAX).unwrap();
            actual.sort_unstable(); assert_eq!(actual,expected);
            for budget in 0..=64 {
                if let Some(mut actual)=tree.token_ids_with_proper_prefix_byte(&query,budget) {
                    actual.sort_unstable(); assert_eq!(actual,expected,"budget must never truncate the answer");
                }
            }
        }
        assert!(tree.token_ids_with_proper_prefix_byte(&mask(*b"b"),0).is_none());
        assert_eq!(tree.token_ids_with_proper_prefix_byte(&[0;4],0),Some(Vec::new()));
    }

    #[test]
    fn proper_prefix_byte_cardinality_budget_excludes_only_the_matching_endpoint() {
        let entries = vec![(7,b"a".to_vec()),(19,b"aa".to_vec()),(41,b"ab".to_vec())];
        let tree = VocabPrefixTree::build_owned(entries.clone());
        // One root visit, one matching compressed edge and two strict
        // descendants. The endpoint "a" is not an extra emitted ID.
        assert_eq!(tree.token_ids_with_proper_prefix_byte(&mask(*b"a"),3),None);
        let mut ids = tree.token_ids_with_proper_prefix_byte(&mask(*b"a"),4).unwrap();
        ids.sort_unstable(); assert_eq!(ids,vec![19,41]);
        let leaf = VocabPrefixTree::build_owned(vec![(99,b"za".to_vec())]);
        assert_eq!(leaf.token_ids_with_proper_prefix_byte(&mask(*b"a"),2),Some(Vec::new()));
        assert_eq!(leaf.token_ids_with_proper_prefix_byte(&mask(*b"z"),2),None);
        assert_eq!(leaf.token_ids_with_proper_prefix_byte(&mask(*b"z"),3),Some(vec![99]));
    }

    #[test]
    fn proper_prefix_byte_dense_query_declines_without_returning_a_partial_subtree() {
        let entries = (0..8192usize).map(|id| {
            let mut bytes=vec![b'a'+u8::from(id>=4096)];
            bytes.extend_from_slice(&(id as u32).to_le_bytes());
            (id*3+11,bytes)
        }).collect::<Vec<_>>();
        let tree=VocabPrefixTree::build_owned(entries.clone());
        let query=mask(*b"ab");
        // Either first subtree fits alone, but their union does not. The
        // complete plan must succeed before any partial answer is returned.
        assert!(tree.token_ids_with_proper_prefix_byte(&query,5000).is_none());
        let mut ids=tree.token_ids_with_proper_prefix_byte(&query,8195).unwrap();
        ids.sort_unstable(); assert_eq!(ids,reference(&entries,&query));
        assert!(tree.token_ids_with_proper_prefix_byte(&query,8194).is_none());
    }

    #[test]
    fn proper_prefix_byte_query_matches_independent_all_byte_and_random_set_scans() {
        fn next(state: &mut u64)->u64 {
            *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17; *state
        }
        let mut random=0x0010_9174_b00d_u64;
        for _case in 0..96 {
            let mut words=BTreeSet::new(); words.insert(Vec::new());
            for _ in 0..64 {
                let len=(next(&mut random)%13) as usize;
                words.insert((0..len).map(|_| next(&mut random) as u8).collect::<Vec<_>>());
            }
            let entries=words.into_iter().enumerate().map(|(i,b)|(i*31+11,b)).collect::<Vec<_>>();
            let tree=VocabPrefixTree::build_owned(entries.clone());
            for byte in 0..=255u8 {
                let query=mask([byte]); let expected=reference(&entries,&query);
                let mut actual=tree.token_ids_with_proper_prefix_byte(&query,usize::MAX).unwrap();
                actual.sort_unstable(); assert_eq!(actual,expected);
            }
            for _ in 0..128 {
                let query=[next(&mut random),next(&mut random),next(&mut random),next(&mut random)];
                let expected=reference(&entries,&query);
                let mut actual=tree.token_ids_with_proper_prefix_byte(&query,usize::MAX).unwrap();
                actual.sort_unstable(); assert_eq!(actual,expected);
            }
        }
    }
}
