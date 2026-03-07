//! Lexer-side NFA → DFA determinization.

use super::dfa::Dfa;
use super::nfa::Nfa;

impl Nfa {
    /// Convert this NFA to a DFA via subset construction.
    ///
    /// Uses input equivalence classes to reduce the alphabet size,
    /// then builds the DFA using the standard powerset/subset construction.
    pub fn to_dfa(&self) -> Dfa {
        unimplemented!()
    }

    /// Standard subset construction NFA → DFA.
    fn subset_construction(&self) -> Dfa {
        unimplemented!()
    }

    /// Compute input equivalence classes.
    ///
    /// Returns `(class_map, num_classes, class_members)` where:
    /// - `class_map[byte]` = class ID for that byte
    /// - `num_classes` = number of distinct classes
    /// - `class_members[class]` = one representative byte for each class
    fn compute_equivalence_classes(&self) -> (Vec<u8>, u8, Vec<u8>) {
        unimplemented!()
    }
}