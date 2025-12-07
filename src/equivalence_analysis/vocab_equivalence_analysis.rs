//! Vocab Equivalence Analysis Dispatcher
//!
//! This module provides the main entry point for vocab equivalence analysis,
//! routing between the fast implementation and optionally validating against
//! the reference implementation when testing.

use crate::finite_automata::Regex;
use hashbrown::HashMap;
use std::collections::BTreeSet;

pub use super::vocab_equivalence_analysis_fast::VocabEquivalenceResult;
use super::{vocab_equivalence_analysis_fast, vocab_equivalence_analysis_reference, vocab_equivalence_trie};

/// Find vocab equivalence classes of tokens based on DFA behavior.
///
/// Two tokens are equivalent if they produce identical parsing behavior
/// across all initial tokenizer states.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `strings` - Vocabulary tokens to analyze
/// * `initial_states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// Sets of token indices that are equivalent (produce identical parsing behavior).
///
/// # Testing
/// Set the environment variable `VOCAB_EQUIVALENCE_ANALYSIS_TEST=1` to enable
/// validation against the reference implementation.
///
/// # Algorithm Selection
/// Set the environment variable `USE_TRIE_VOCAB_EQUIV=1` to use the trie-based
/// algorithm instead of the batch-based algorithm.
pub fn find_vocab_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    // Check if trie-based algorithm is requested
    let use_trie = std::env::var("USE_TRIE_VOCAB_EQUIV").is_ok();
    
    // Skip validation unless explicitly requested via ENV var
    if std::env::var("VOCAB_EQUIVALENCE_ANALYSIS_TEST").is_ok() {
        let instant = std::time::Instant::now();
        let reference =
            vocab_equivalence_analysis_reference::find_vocab_equivalence_classes(regex, strings, initial_states);
        crate::debug!(
            3,
            "Reference vocab equivalence analysis took {:?}",
            instant.elapsed()
        );
        let instant = std::time::Instant::now();
        let fast = vocab_equivalence_analysis_fast::find_vocab_equivalence_classes(regex, strings, initial_states);
        crate::debug!(3, "Fast vocab equivalence analysis took {:?}", instant.elapsed());
        
        if reference != fast {
            fn build_maps(groups: &VocabEquivalenceResult) -> (HashMap<usize, usize>, HashMap<usize, Vec<usize>>) {
                let mut idx_to_rep = HashMap::new();
                let mut rep_to_group = HashMap::new();
                for g in groups {
                    if let Some(&rep) = g.first() {
                        rep_to_group.insert(rep, g.clone());
                        for &idx in g {
                            idx_to_rep.insert(idx, rep);
                        }
                    }
                }
                (idx_to_rep, rep_to_group)
            }

            let (ref_map, _) = build_maps(&reference);
            let (fast_map, _) = build_maps(&fast);

            eprintln!(
                "Vocab equivalence mismatch: reference groups {} fast groups {}",
                reference.len(),
                fast.len()
            );

            let mut mismatches = 0;
            for idx in 0..strings.len() {
                let r = ref_map.get(&idx);
                let f = fast_map.get(&idx);
                if r != f {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!("idx {} ref_rep {:?} fast_rep {:?}", idx, r, f);
                    }
                }
            }
            if mismatches > 5 {
                eprintln!("... and {} more mismatches", mismatches - 5);
            }

            panic!("Mismatch between reference and fast vocab equivalence analysis results");
        }
        return fast;
    }

    // Use trie-based algorithm if requested
    if use_trie {
        let groups = vocab_equivalence_trie::find_vocab_equivalence_classes_trie(regex, strings, initial_states);
        return groups.into_iter().collect();
    }

    // Default: use fast implementation (state reduction should be done by caller)
    vocab_equivalence_analysis_fast::find_vocab_equivalence_classes(regex, strings, initial_states)
}
