use crate::finite_automata::{GroupID, Regex};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

// A canonical representation of a signature. It can be hashed and compared.
// It's derived from the graph of tokenization possibilities.
// Using BTreeSet to ensure canonical order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalSignature {
    // For each possible match starting at the current position, we store the group ID,
    // the width of the match, and the signature ID of the remaining string.
    matches: BTreeSet<(GroupID, usize, SignatureId)>,
    // If the regex can terminate after this string, this holds the signature of
    // what comes next (which is effectively the signature for an empty string from
    // the next tokenizer state, which is always the initial state).
    final_signature_if_done: Option<SignatureId>,
    // Signature of the string remainder after consuming one byte.
    next_byte_signature: Option<SignatureId>,
}

// An ID representing a unique canonical signature.
pub type SignatureId = usize;

// A helper struct to manage interning of signatures.
struct SignatureInterner {
    signatures: Vec<CanonicalSignature>,
    map: HashMap<CanonicalSignature, SignatureId>,
    empty_signature_id: SignatureId,
}

impl SignatureInterner {
    fn new() -> Self {
        let mut interner = Self {
            signatures: Vec::new(),
            map: HashMap::new(),
            empty_signature_id: 0,
        };
        // The signature for an empty string from the initial state.
        // It has no matches and doesn't consume bytes.
        let empty_sig = CanonicalSignature {
            matches: BTreeSet::new(),
            final_signature_if_done: None,
            next_byte_signature: None,
        };
        interner.empty_signature_id = interner.intern(empty_sig);
        interner
    }
    fn intern(&mut self, sig: CanonicalSignature) -> SignatureId {
        if let Some(&id) = self.map.get(&sig) {
            return id;
        }
        let id = self.signatures.len();
        self.signatures.push(sig.clone());
        self.map.insert(sig, id);
        id
    }
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u8, usize>,
    string_indices: Vec<usize>,
}

// A Trie to store all input strings and facilitate shared computation.
struct StringTrie {
    nodes: Vec<TrieNode>,
}

impl StringTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }
    fn insert(&mut self, s: &[u8], index: usize) {
        let mut current_node_id = 0;
        for &byte in s {
            let current_node = &mut self.nodes[current_node_id];
            current_node_id = *current_node.children.entry(byte).or_insert_with(|| {
                self.nodes.push(TrieNode::default());
                self.nodes.len() - 1
            });
        }
        self.nodes[current_node_id].string_indices.push(index);
    }

    fn post_order_traversal(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![(0, false)]; // (node_id, children_visited)
        let mut visited = BTreeSet::new();

        while let Some((node_id, children_visited)) = stack.pop() {
            if children_visited {
                result.push(node_id);
            } else {
                if visited.contains(&node_id) {
                    continue;
                }
                visited.insert(node_id);
                stack.push((node_id, true));
                let node = &self.nodes[node_id];
                // Iterate in a deterministic order for reproducibility
                let mut children: Vec<_> = node.children.values().collect();
                children.sort();
                for &child_id in children {
                    stack.push((child_id, false));
                }
            }
        }
        result
    }
}

// Represents the result of running `regex.execute(string)`.
#[derive(Debug, Clone)]
struct ExecResult {
    matches: BTreeMap<GroupID, usize>,
    final_state: Option<usize>,
}

pub struct EquivalenceAnalyzer<'a> {
    regex: &'a Regex,
    strings: &'a [Vec<u8>],
    initial_states: &'a [usize],
    trie: StringTrie,
    signature_memo: HashMap<(usize, usize), SignatureId>, // (trie_node_id, dfa_state) -> SignatureId
    signature_interner: SignatureInterner,
}

impl<'a> EquivalenceAnalyzer<'a> {
    pub fn new(regex: &'a Regex, strings: &'a [Vec<u8>], initial_states: &'a [usize]) -> Self {
        Self {
            regex,
            strings,
            initial_states,
            trie: StringTrie::new(),
            signature_memo: HashMap::new(),
            signature_interner: SignatureInterner::new(),
        }
    }

    pub fn find_equivalence_classes(&mut self) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
        self.build_trie();
        self.compute_signatures();
        self.classify_strings()
    }

    fn build_trie(&mut self) {
        for (i, s) in self.strings.iter().enumerate() {
            self.trie.insert(s, i);
        }
    }

    fn compute_signatures(&mut self) {
        let post_order = self.trie.post_order_traversal();
        let all_states: Vec<_> = self.regex.iter_states().collect();

        for &node_id in &post_order {
            let node = &self.trie.nodes[node_id];
            let prefix = self.get_prefix_for_node(node_id);

            for &dfa_state in &all_states {
                let key = (node_id, dfa_state.0);
                if self.signature_memo.contains_key(&key) {
                    continue;
                }

                let mut sig_matches = BTreeSet::new();
                let res = self.regex.execute_from_state(&prefix, dfa_state);

                for m in res.matches {
                    let remaining_prefix = &prefix[m.width..];
                    let (rem_node_id, rem_prefix_in_node) = self.find_node_for_prefix(node_id, remaining_prefix);
                    assert!(rem_prefix_in_node.is_empty()); // Match must align with trie nodes

                    // After a match, the tokenizer resets to its initial state for the remainder.
                    let rem_sig_id = *self.signature_memo.get(&(rem_node_id, self.regex.initial_state_id().0)).unwrap();
                    sig_matches.insert((m.id, m.width, rem_sig_id));
                }

                let final_sig = if res.end_state.is_some() {
                    Some(self.signature_interner.empty_signature_id)
                } else {
                    None
                };

                let next_byte_sig = if !prefix.is_empty() {
                    let rem_node_id = self.find_node_for_prefix(0, &prefix[1..]).0;
                    let next_dfa_state = self.regex.execute_from_state(&[prefix[0]], dfa_state).end_state;
                    next_dfa_state.and_then(|s| self.signature_memo.get(&(rem_node_id, s)))
                        .copied()
                } else {
                    None
                };

                let sig = CanonicalSignature {
                    matches: sig_matches,
                    final_signature_if_done: final_sig,
                    next_byte_signature: next_byte_sig,
                };
                let sig_id = self.signature_interner.intern(sig);
                self.signature_memo.insert(key, sig_id);
            }
        }
    }

    fn classify_strings(&self) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
        let mut classes = BTreeMap::new();
        for (i, s) in self.strings.iter().enumerate() {
            let (node_id, rem) = self.find_node_for_prefix(0, s);
            assert!(rem.is_empty());

            let mut sig_vec = Vec::with_capacity(self.initial_states.len());
            for &initial_state in self.initial_states {
                let sig_id = *self.signature_memo.get(&(node_id, initial_state)).unwrap();
                sig_vec.push(sig_id);
            }
            classes.entry(sig_vec).or_insert_with(Vec::new).push(i);
        }
        classes
    }

    fn find_node_for_prefix(&self, start_node_id: usize, prefix: &[u8]) -> (usize, &[u8]) {
        let mut current_node_id = start_node_id;
        for (i, &byte) in prefix.iter().enumerate() {
            if let Some(&child_id) = self.trie.nodes[current_node_id].children.get(&byte) {
                current_node_id = child_id;
            } else {
                return (current_node_id, &prefix[i..]);
            }
        }
        (current_node_id, &[])
    }

    fn get_prefix_for_node(&self, node_id: usize) -> Vec<u8> {
        if node_id == 0 {
            return Vec::new();
        }
        let mut path: Vec<u8> = Vec::new();
        let mut q = VecDeque::from([(0, Vec::new())]);
        let mut visited = BTreeSet::from([0]);
        while let Some((curr_id, curr_path)) = q.pop_front() {
            if curr_id == node_id {
                return curr_path;
            }
            let node = &self.trie.nodes[curr_id];
            for (&byte, &child_id) in &node.children {
                if visited.insert(child_id) {
                    let mut next_path = curr_path.clone();
                    next_path.push(byte);
                    q.push_back((child_id, next_path));
                }
            }
        }
        panic!("Node ID not found in trie");
    }
}

// The public-facing function.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
    let mut analyzer = EquivalenceAnalyzer::new(regex, strings, initial_states);
    analyzer.find_equivalence_classes()
}
