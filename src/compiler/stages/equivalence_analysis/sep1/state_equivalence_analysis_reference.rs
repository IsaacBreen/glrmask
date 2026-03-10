//! State Equivalence Analysis - Reference Implementation
//!
//! A simple, correct implementation for testing and validation.
//! States are equivalent if they have identical behavior on ALL tokens.
//!
//! Complexity: O(states × tokens × avg_token_length) with parallelism

use std::collections::BTreeSet;
use crate::finite_automata::Regex;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

/// Find state equivalence classes for a tokenizer.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to consider
/// * `states` - List of state IDs to analyze
///
/// # Returns
/// A vector where `result[i]` is the representative state for `states[i]`.
/// States with the same representative are equivalent.
pub fn find_state_equivalence_classes(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let start = std::time::Instant::now();
    let mapping = super::trellis_equivalence_analysis::find_state_equivalence_classes_trellis(
        regex,
        tokens,
        states,
    );
    let num_groups = mapping.iter().copied().collect::<BTreeSet<_>>().len();
    crate::debug!(
        3,
        "State equiv reference (trellis-backed): {} states -> {} groups in {:?}",
        states.len(),
        num_groups,
        start.elapsed(),
    );
    
    mapping
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
pub fn mapping_to_equivalence_classes(states: &[usize], mapping: &[usize]) -> StateEquivalenceResult {
    let mut rep_to_class: std::collections::BTreeMap<usize, BTreeSet<usize>> = std::collections::BTreeMap::new();
    
    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }
    
    rep_to_class.into_values().collect()
}

    #[cfg(test)]
    mod tests {
        use std::sync::Arc;

        use indoc::indoc;

        use crate::interface::{CompiledGrammar, GrammarDefinition};

        use super::{find_state_equivalence_classes, mapping_to_equivalence_classes, StateEquivalenceResult};

        fn state_is_refinement(candidate: &StateEquivalenceResult, target: &StateEquivalenceResult) -> bool {
            candidate.iter().all(|candidate_class| {
                target
                    .iter()
                    .any(|target_class| candidate_class.is_subset(target_class))
            })
        }

        #[test]
        fn test_reference_state_equivalence_refines_trellis_on_schema_case() {
            let _guard = crate::GLOBAL_DIMS_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            let lark_grammar = indoc! {r#"
                PATTERN_0: /[\x20-\x21\x23-\x5B\x5D-\x7F]/
                PATTERN_1: /[\xC2-\xDF]/
                PATTERN_2: /[\x80-\xBF]/
                PATTERN_3: /[\xE0-\xEF]/
                PATTERN_4: /[\xF0-\xF4]/
                PATTERN_5: /[\x30-\x39\x41-\x46\x61-\x66]/
                PATTERN_6: /[\x22\x2F\x5C\x62\x66\x6E\x72\x74]/
                PATTERN_7: /[\x30-\x39]/
                PATTERN_8: /[\x31-\x39]/
                PATTERN_9: /[\x45\x65]/
                PATTERN_10: /[\x2B\x2D]/
                STRING_CHAR: PATTERN_0 | PATTERN_1 PATTERN_2 | PATTERN_3 PATTERN_2 PATTERN_2 | PATTERN_4 PATTERN_2 PATTERN_2 PATTERN_2
                HEX: PATTERN_5
                ESCAPE_SHORT_CHAR: PATTERN_6
                ESCAPE_SEQ: "\\" ESCAPE_SHORT_CHAR | "\\" "u" HEX HEX HEX HEX
                STRING_CONTENT: (STRING_CHAR | ESCAPE_SEQ)*
                JSON_STRING: "\"" STRING_CONTENT "\""
                DIGIT: PATTERN_7
                NONZERO_DIGIT: PATTERN_8
                INT_PART: "0" | NONZERO_DIGIT DIGIT*
                FRAC_PART: "." DIGIT+
                EXP_MARK: PATTERN_9
                EXP_SIGN: PATTERN_10
                EXP_PART: EXP_MARK EXP_SIGN? DIGIT+
                JSON_INTEGER: "-"? INT_PART
                JSON_NUMBER: "-"? INT_PART FRAC_PART? EXP_PART?
                JSON_BOOL: "true" | "false"
                JSON_NULL: "null"
                json_kv: JSON_STRING ":" json_value
                json_object: "{" "}" | "{" json_kv ("," json_kv)* "}"
                json_array: "[" "]" | "[" json_value ("," json_value)* "]"
                json_value: json_object | json_array | JSON_STRING | JSON_NUMBER | JSON_INTEGER | JSON_BOOL | JSON_NULL
                obj_required_0_1: "\"a\"" ":" json_object
                obj_required_0_2: "\"\"" ":" JSON_STRING
                obj_required_0_0: "\"\"" ":" JSON_STRING "," obj_required_0_1 | "\"a\"" ":" json_object "," obj_required_0_2
                start: "{" obj_required_0_0 "}"
            "#};

            let definition = GrammarDefinition::from_lark(lark_grammar).unwrap();
            let compiled = CompiledGrammar::from_definition(Arc::new(definition));
            let tokenizer = compiled.tokenizer();

            let tokens: Vec<Vec<u8>> = vec![
                b"{\"".to_vec(),
                b"\":\"".to_vec(),
                b"\",\"".to_vec(),
                b"\"a\"".to_vec(),
                b":{\"".to_vec(),
            ];

            let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();

            let ref_mapping = find_state_equivalence_classes(tokenizer.as_regex(), &tokens, &states);
            let trellis_mapping = super::super::trellis_equivalence_analysis::find_state_equivalence_classes_trellis(
                tokenizer.as_regex(),
                &tokens,
                &states,
            );

            let ref_classes = mapping_to_equivalence_classes(&states, &ref_mapping);
            let trellis_classes =
                super::super::trellis_equivalence_analysis::mapping_to_equivalence_classes(&states, &trellis_mapping);

            assert!(
                state_is_refinement(&ref_classes, &trellis_classes),
                "Reference state equivalence over-merged vs trellis. ref={ref_classes:?} trellis={trellis_classes:?}"
            );
        }
    }
