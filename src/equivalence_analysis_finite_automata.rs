use crate::finite_automata::{GroupID, Regex};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// A canonical representation of a signature. It can be hashed and compared.
// It's derived from the graph of tokenization possibilities.
// It represents the result of a single greedy tokenization step.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalSignature {
    // If a greedy match is found, we store its group, position,
    // and the signature ID of the remainder of the string.
    match_and_remainder: Option<(GroupID, usize, SignatureId)>,
    // If no greedy match is found, we store the final DFA state the regex ended in.
    final_state_if_no_match: Option<usize>,
}

// An ID representing a unique canonical signature.
type SignatureId = usize;

// Manages interning of canonical signatures to unique IDs.
struct SignatureInterner {
    signatures: Vec<CanonicalSignature>,
    map: HashMap<CanonicalSignature, SignatureId>,
}

impl SignatureInterner {
    fn new() -> Self {
        let mut interner = SignatureInterner {
            signatures: Vec::new(),
            map: HashMap::new(),
        };
        // Pre-intern a signature for the empty string case.
        interner.intern(CanonicalSignature {
            match_and_remainder: None,
            final_state_if_no_match: Some(0), // Assuming start_state is 0
        });
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

pub struct EquivalenceAnalyzer<'a> {
    regex: &'a Regex,
    strings: &'a [Vec<u8>],
    initial_states: &'a [usize],
}

impl<'a> EquivalenceAnalyzer<'a> {
    pub fn new(regex: &'a Regex, strings: &'a [Vec<u8>], initial_states: &'a [usize]) -> Self {
        EquivalenceAnalyzer {
            regex,
            strings,
            initial_states,
        }
    }

    pub fn find_equivalence_classes(&mut self) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
        // 1. Collect all unique prefixes from the input strings.
        //    This exploits shared work for common prefixes.
        let mut prefixes = HashSet::new();
        prefixes.insert(Vec::new()); // Base case for recursion
        for s in self.strings {
            for i in 0..=s.len() {
                prefixes.insert(s[0..i].to_vec());
            }
        }
        let mut sorted_prefixes: Vec<Vec<u8>> = prefixes.into_iter().collect();
        sorted_prefixes.sort_by_key(|p| p.len());

        // 2. Identify all DFA states for which we need to compute signatures.
        let relevant_states: BTreeSet<usize> = self.initial_states
            .iter()
            .cloned()
            .chain(std::iter::once(self.regex.dfa.start_state))
            .collect();

        // 3. Compute signatures for all prefixes, from shortest to longest.
        let mut signature_interner = SignatureInterner::new();
        let mut memo: HashMap<Vec<u8>, BTreeMap<usize, SignatureId>> = HashMap::new();

        for prefix in &sorted_prefixes {
            let mut state_sigs = BTreeMap::new();
            for &dfa_state in &relevant_states {
                let mut rs = self.regex.init_to_state(dfa_state);
                rs.execute(prefix);

                let sig = if let Some(m) = rs.get_greedy_match() {
                    let remainder = &prefix[m.position..];
                    let remainder_sigs = memo.get(remainder)
                        .expect("BUG: remainder signature should be pre-computed");
                    let remainder_sig_id = remainder_sigs.get(&self.regex.dfa.start_state)
                        .expect("BUG: remainder signature for start_state should exist");

                    CanonicalSignature {
                        match_and_remainder: Some((m.group_id, m.position, *remainder_sig_id)),
                        final_state_if_no_match: None,
                    }
                } else {
                    CanonicalSignature {
                        match_and_remainder: None,
                        final_state_if_no_match: Some(rs.current_state),
                    }
                };
                state_sigs.insert(dfa_state, signature_interner.intern(sig));
            }
            memo.insert(prefix.clone(), state_sigs);
        }

        // 4. Classify original strings based on their signature vectors.
        let mut equivalence_classes: BTreeMap<Vec<SignatureId>, Vec<usize>> = BTreeMap::new();
        for (i, s) in self.strings.iter().enumerate() {
            let mut signature_vector = Vec::with_capacity(self.initial_states.len());
            if let Some(string_sigs) = memo.get(s) {
                for &initial_state in self.initial_states {
                    let sig_id = string_sigs.get(&initial_state)
                        .expect("BUG: Signature for initial state not found");
                    signature_vector.push(*sig_id);
                }
            }
            equivalence_classes.entry(signature_vector).or_default().push(i);
        }

        equivalence_classes
    }
}

/// Finds equivalence classes among a set of strings based on their tokenization
/// behavior with a given Regex, starting from a set of initial DFA states.
///
/// Two strings are considered equivalent if, for every initial DFA state provided,
/// they produce the same sequence of tokens.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BTreeMap<Vec<SignatureId>, Vec<usize>> {
    let mut analyzer = EquivalenceAnalyzer::new(regex, strings, initial_states);
    analyzer.find_equivalence_classes()
}
