use crate::finite_automata::{GroupID, Match, Regex};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

// For debugging: verify equivalence classes using a brute-force method.
const VERIFY_EQUIVALENCE_CLASSES: bool = true;

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
        // 1. Collect all unique substrings from the input strings.
        //    We need all substrings because a remainder after a match can be any substring.
        let mut substrings = HashSet::new();
        substrings.insert(Vec::new()); // Base case for recursion
        for s in self.strings {
            for i in 0..s.len() {
                for j in i..=s.len() {
                    substrings.insert(s[i..j].to_vec());
                }
            }
        }
        let mut sorted_substrings: Vec<Vec<u8>> = substrings.into_iter().collect();
        sorted_substrings.sort_by_key(|p| p.len());

        crate::debug!(2, "Starting LLM token equivalence analysis for {} unique substrings...", sorted_substrings.len());
        let pb = ProgressBar::new(sorted_substrings.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta}) (Equivalence Analysis)")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        // 2. Identify all DFA states for which we need to compute signatures.
        let relevant_states: BTreeSet<usize> = self.initial_states
            .iter()
            .cloned()
            .chain(std::iter::once(self.regex.dfa.start_state))
            .collect();

        // 3. Compute signatures for all prefixes, from shortest to longest.
        let mut signature_interner = SignatureInterner::new();
        let mut memo: HashMap<Vec<u8>, BTreeMap<usize, SignatureId>> = HashMap::new();

        for substring in &sorted_substrings {
            pb.inc(1);
            let mut state_sigs = BTreeMap::new();
            for &dfa_state in &relevant_states {
                let mut rs = self.regex.init_to_state(dfa_state);
                rs.execute(substring);

                let sig = if let Some(m) = rs.get_greedy_match() {
                    let remainder = &substring[m.position..];
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
            memo.insert(substring.clone(), state_sigs);
        }
        pb.finish();

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

        if VERIFY_EQUIVALENCE_CLASSES {
            verify_equivalence_classes(self.regex, self.strings, self.initial_states, &equivalence_classes);
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

/// Brute-force verification of equivalence classes.
/// This is slow and should only be used for debugging.
fn verify_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
    computed_classes: &BTreeMap<Vec<SignatureId>, Vec<usize>>,
) {
    println!("Verifying equivalence classes (this may be slow)...");

    let mut brute_force_classes: BTreeMap<Vec<Vec<Match>>, Vec<usize>> = BTreeMap::new();
    for (i, s) in strings.iter().enumerate() {
        let mut signature_vector = Vec::with_capacity(initial_states.len());
        for &initial_state in initial_states {
            let mut rs = regex.init_to_state(initial_state);
            let matches = rs.greedy_find_all(s, true);
            signature_vector.push(matches);
        }
        brute_force_classes.entry(signature_vector).or_default().push(i);
    }

    let mut computed_partitions: Vec<BTreeSet<usize>> = computed_classes
        .values()
        .map(|v| v.iter().cloned().collect())
        .collect();
    computed_partitions.sort();

    let mut brute_force_partitions: Vec<BTreeSet<usize>> = brute_force_classes
        .values()
        .map(|v| v.iter().cloned().collect())
        .collect();
    brute_force_partitions.sort();

    if computed_partitions == brute_force_partitions {
        println!("Equivalence class verification successful!");
    } else {
        eprintln!("Equivalence class verification FAILED!");
        eprintln!("Computed partitions ({}): {:?}", computed_partitions.len(), computed_partitions);
        eprintln!("Brute-force partitions ({}): {:?}", brute_force_partitions.len(), brute_force_partitions);
        panic!("Equivalence class verification failed.");
    }
}
