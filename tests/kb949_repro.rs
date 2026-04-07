/// Regression test for kb_949: token `.k` was incorrectly merged into a
/// catch-all equivalence class when the DFA state count crossed a threshold
/// that triggered `pre_vocab_state_reduction` with an unsound early_stop.
///
/// The root cause: `find_state_equivalence_classes_ex_with_rep_confirmation`
/// could declare convergence after processing only a fraction of the token
/// list, missing a distinguishing token in a later batch. This caused states
/// that differ for rare tokens to be merged, which propagated to incorrect
/// vocab equivalence classes.

use glrmask::{Constraint, Vocab};

/// Build a vocab with enough tokens to trigger `pre_vocab_state_reduction`.
/// Includes all single-byte ASCII tokens plus many 2-byte and 3-byte tokens
/// to cross the dedup threshold.
fn build_test_vocab() -> (Vocab, u32, u32) {
    let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut next_id = 0u32;

    // All single-byte tokens (0x00..0xFF)
    for b in 0..=255u8 {
        entries.push((next_id, vec![b]));
        next_id += 1;
    }

    let dot_k_id;
    let dot_s_id;

    // 2-byte tokens: all combinations of first-byte × second-byte for printable ASCII
    // This creates enough diversity to cross thresholds.
    for b1 in 0x20..0x7Fu8 {
        for b2 in 0x20..0x7Fu8 {
            // Track .k and .s specifically
            entries.push((next_id, vec![b1, b2]));
            next_id += 1;
        }
    }

    // 3-byte tokens: subset of printable ASCII combinations
    for b1 in b"\":{},[]".iter() {
        for b2 in 0x20..0x7Fu8 {
            for b3 in 0x20..0x7Fu8 {
                entries.push((next_id, vec![*b1, b2, b3]));
                next_id += 1;
            }
        }
    }

    // 4-byte tokens for common JSON patterns
    for prefix in &[b"null", b"true", b"fals"] {
        entries.push((next_id, prefix.to_vec()));
        next_id += 1;
    }
    entries.push((next_id, b"false".to_vec()));
    next_id += 1;

    // Find .k and .s token IDs
    dot_k_id = entries
        .iter()
        .find(|(_, b)| b == b".k")
        .map(|(id, _)| *id)
        .expect(".k token not found");
    dot_s_id = entries
        .iter()
        .find(|(_, b)| b == b".s")
        .map(|(id, _)| *id)
        .expect(".s token not found");

    (Vocab::new(entries, None), dot_k_id, dot_s_id)
}

#[test]
fn test_kb949_dot_k_not_merged_with_dot_s() {
    // Force pre_vocab_state_reduction even if thresholds aren't met.
    // The bug only manifests when this reduction path is active.
    unsafe { std::env::set_var("GLRMASK_FORCE_PRE_VOCAB_STATE_REDUCTION", "1") };

    let (vocab, dot_k_id, dot_s_id) = build_test_vocab();

    // Schema with enough complexity to generate >200 DFA states.
    // The n=9 properties in array items are the trigger.
    let n = 9;
    let props: String = (0..n)
        .map(|i| format!("\"prop{}\": {{\"type\": \"string\"}}", i))
        .collect::<Vec<_>>()
        .join(", ");

    let schema = format!(
        r#"{{
            "type": "object",
            "properties": {{
                "apiVersion": {{"type": ["string", "null"], "enum": ["authorization.k8s.io/v1"]}},
                "kind": {{"type": ["string", "null"], "enum": ["SelfSubjectAccessReview"]}},
                "metadata": {{
                    "type": "object",
                    "properties": {{
                        "deletionGracePeriodSeconds": {{"type": "integer"}},
                        "managedFields": {{"type": "array", "items": {{
                            "type": "object",
                            "properties": {{ {props} }}
                        }}}},
                        "ownerReferences": {{"type": "array", "items": {{
                            "type": "object",
                            "properties": {{ {props} }}
                        }}}},
                        "selfLink": {{"type": "string"}}
                    }}
                }}
            }}
        }}"#
    );

    let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();

    // Clean up env var
    unsafe { std::env::remove_var("GLRMASK_FORCE_PRE_VOCAB_STATE_REDUCTION") };

    // Core assertion: .k and .s must be in different equivalence classes.
    // Before the fix, they were incorrectly merged into a 108K-token class.
    let token_map = constraint.debug_original_token_to_internal();
    let dot_k_class = token_map[dot_k_id as usize];
    let dot_s_class = token_map[dot_s_id as usize];

    assert_ne!(
        dot_k_class, dot_s_class,
        "Token .k (id={}) and .s (id={}) must be in different equivalence classes \
         (both mapped to class {}). This indicates the pre_vocab_state_reduction \
         early_stop bug has regressed.",
        dot_k_id, dot_s_id, dot_k_class,
    );
}
