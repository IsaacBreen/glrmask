use crate::finite_automata::Regex;
use crate::{equivalence_analysis_fast, equivalence_analysis_reference};
use hashbrown::HashMap;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Skip validation unless explicitly requested via ENV var
    if std::env::var("EQUIVALENCE_ANALYSIS_TEST").is_ok() {
        let instant = std::time::Instant::now();
        let reference =
            equivalence_analysis_reference::find_equivalence_classes(regex, strings, initial_states);
        crate::debug!(
            3,
            "Reference equivalence analysis took {:?}",
            instant.elapsed()
        );
        let instant = std::time::Instant::now();
        let fast = equivalence_analysis_fast::find_equivalence_classes(regex, strings, initial_states);
        crate::debug!(3, "Fast equivalence analysis took {:?}", instant.elapsed());
        
        if reference != fast {
            fn build_maps(groups: &EquivalenceResult) -> (HashMap<usize, usize>, HashMap<usize, Vec<usize>>) {
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
                "Equivalence mismatch: reference groups {} fast groups {}",
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

            panic!("Mismatch between reference and fast equivalence analysis results");
        }
        return fast;
    }

    // Default: use fast implementation (state reduction is done by caller in constraint.rs)
    equivalence_analysis_fast::find_equivalence_classes(regex, strings, initial_states)
}
