use std::sync::OnceLock;

/// Exact suffix identity is the reverse lexical trie. Nodes at one length
/// occupy one contiguous interval, and every tail precedes its longer
/// suffixes. L1 reverse subset scans and L2+ recursive suffix semantics can
/// consume the same topology without either owning another spelling copy.
#[derive(Debug)]
pub struct VocabularySuffixIndex {
    pub parents: Box<[u32]>,
    pub bytes: Box<[u8]>,
    pub token_nodes: Box<[u32]>,
    pub level_offsets: Box<[usize]>,
    positions: OnceLock<SuffixPositions>,
}

#[derive(Debug)]
pub struct SuffixPositions {
    offsets: Box<[usize]>,
    ids: Box<[u32]>,
    representatives: Box<[(usize, usize)]>,
}

impl SuffixPositions {
    #[inline]
    pub fn suffix_id(&self, token: usize, offset: usize) -> u32 {
        let start = self.offsets[token];
        assert!(offset < self.offsets[token + 1] - start);
        self.ids[start + offset]
    }

    #[inline]
    pub fn representative(&self, id: usize) -> (usize, usize) {
        self.representatives[id]
    }
}

impl VocabularySuffixIndex {
    pub fn build<S: AsRef<[u8]>>(tokens: &[S]) -> Self {
        let mut order = (0..tokens.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|&left, &right| {
            tokens[left].as_ref().iter().rev().cmp(tokens[right].as_ref().iter().rev())
                .then(left.cmp(&right))
        });
        Self::build_reverse_sorted(tokens, order)
    }

    /// The supplied order is a permutation of token positions sorted by
    /// reversed bytes. Parallel callers may prepare that order themselves.
    pub fn build_reverse_sorted<S: AsRef<[u8]>>(tokens: &[S], order: Vec<usize>) -> Self {
        let mut parents = vec![u32::MAX];
        let mut bytes = vec![0u8];
        let mut depths = vec![0usize];
        let mut token_nodes = vec![u32::MAX; tokens.len()];
        let mut path = vec![0u32];
        let mut previous = None::<usize>;
        let mut level_counts = vec![1usize];
        for token in order {
            let spelling = tokens[token].as_ref();
            let lcp = previous.map_or(0, |prev| {
                tokens[prev].as_ref().iter().rev().zip(spelling.iter().rev())
                    .take_while(|(a, b)| a == b).count()
            });
            path.truncate(lcp + 1);
            let mut parent = path[lcp];
            for &byte in spelling[..spelling.len() - lcp].iter().rev() {
                let node = u32::try_from(parents.len()).expect("suffix node index");
                assert_ne!(node, u32::MAX, "suffix index exhausted");
                let depth = path.len();
                if depth >= level_counts.len() { level_counts.resize(depth + 1, 0); }
                level_counts[depth] += 1;
                parents.push(parent);
                bytes.push(byte);
                depths.push(depth);
                path.push(node);
                parent = node;
            }
            debug_assert_eq!(token_nodes[token], u32::MAX);
            token_nodes[token] = parent;
            previous = Some(token);
        }
        // Breadth-first layout keeps the reverse subset scan compact and
        // permits a length-ordered DP without an extra per-node length array.
        let mut level_offsets = Vec::with_capacity(level_counts.len() + 1);
        level_offsets.push(0usize);
        for count in level_counts {
            level_offsets.push(level_offsets.last().copied().unwrap() + count);
        }
        let mut cursors = level_offsets[..level_offsets.len() - 1].to_vec();
        let mut old_to_new = vec![0u32; parents.len()];
        for (old, depth) in depths.into_iter().enumerate() {
            old_to_new[old] = cursors[depth] as u32;
            cursors[depth] += 1;
        }
        let mut reordered_parents = vec![u32::MAX; parents.len()];
        let mut reordered_bytes = vec![0u8; parents.len()];
        for old in 1..parents.len() {
            let new = old_to_new[old] as usize;
            reordered_parents[new] = old_to_new[parents[old] as usize];
            reordered_bytes[new] = bytes[old];
        }
        for node in &mut token_nodes { *node = old_to_new[*node as usize]; }
        Self {
            parents: reordered_parents.into_boxed_slice(),
            bytes: reordered_bytes.into_boxed_slice(),
            token_nodes: token_nodes.into_boxed_slice(),
            level_offsets: level_offsets.into_boxed_slice(),
            positions: OnceLock::new(),
        }
    }

    /// The position projection is only materialized for consumers that need
    /// random cuts. Reverse token classification needs only the core trie.
    pub fn positions(&self) -> &SuffixPositions {
        self.positions.get_or_init(|| {
            let mut offsets = Vec::with_capacity(self.token_nodes.len() + 1);
            let mut ids = Vec::new();
            let mut representatives = vec![(usize::MAX, 0); self.parents.len()];
            offsets.push(0);
            for (token, &node) in self.token_nodes.iter().enumerate() {
                let mut node = node;
                let mut offset = 0;
                loop {
                    ids.push(node);
                    let representative = &mut representatives[node as usize];
                    if representative.0 == usize::MAX { *representative = (token, offset); }
                    if node == 0 { break; }
                    node = self.parents[node as usize];
                    offset += 1;
                }
                offsets.push(ids.len());
            }
            SuffixPositions {
                offsets: offsets.into_boxed_slice(),
                ids: ids.into_boxed_slice(),
                representatives: representatives.into_boxed_slice(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::VocabularySuffixIndex;

    #[test]
    fn suffix_identity_matches_literal_bytes_and_parent_action() {
        let words = [b"".as_slice(), b"cab", b"ab", b"xxab", b"b", b"cab", b"x"];
        let index = VocabularySuffixIndex::build(&words);
        assert!(index.positions.get().is_none());
        let positions = index.positions();
        for (i, left) in words.iter().enumerate() {
            for cut in 0..=left.len() {
                let id = positions.suffix_id(i, cut);
                let (representative, offset) = positions.representative(id as usize);
                assert_eq!(&words[representative][offset..], &left[cut..]);
                if cut < left.len() {
                    assert_eq!(index.bytes[id as usize], left[cut]);
                    assert_eq!(index.parents[id as usize], positions.suffix_id(i, cut + 1));
                    assert!(index.parents[id as usize] < id);
                }
                for (j, right) in words.iter().enumerate() {
                    for right_cut in 0..=right.len() {
                        assert_eq!(id == positions.suffix_id(j, right_cut), left[cut..] == right[right_cut..]);
                    }
                }
            }
        }
    }
}
