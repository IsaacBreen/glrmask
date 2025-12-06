use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use crate::{equivalence_analysis_fast, equivalence_analysis_reference};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    if std::env::var("SKIP_EQUIVALENCE_ANALYSIS_TEST").is_ok() {
        let fast = equivalence_analysis_fast::find_equivalence_classes(regex, strings, initial_states);
        return fast;
    }
    let instant = std::time::Instant::now();
    let reference = equivalence_analysis_reference::find_equivalence_classes(regex, strings, initial_states);
    crate::debug!(3, "Reference equivalence analysis took {:?}", instant.elapsed());
    let instant = std::time::Instant::now();
    let fast = equivalence_analysis_fast::find_equivalence_classes(regex, strings, initial_states);
    crate::debug!(3, "Fast equivalence analysis took {:?}", instant.elapsed());
    assert_eq!(reference, fast, "Mismatch between reference and fast equivalence analysis results");
    reference
}