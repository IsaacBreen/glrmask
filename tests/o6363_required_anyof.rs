/// Regression test for o6363: `required` keyword was not listed in
/// `has_structural_keywords`, so when a schema had `required: ["image"]`
/// alongside `anyOf` (but no `properties` or `type` at the base level),
/// `convert_structural_branches` skipped the base-variant merge.
/// This dropped the `required` constraint entirely, allowing property-only
/// variants (e.g. `kaniko` without `image`) that should have been rejected.

use glrmask::{Constraint, Vocab};

fn build_small_vocab() -> Vocab {
    let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut next_id = 0u32;

    // All single-byte tokens (0x00..0xFF)
    for b in 0..=255u8 {
        entries.push((next_id, vec![b]));
        next_id += 1;
    }

    // 2-byte tokens for common JSON patterns
    for b1 in 0x20..0x7Fu8 {
        for b2 in 0x20..0x7Fu8 {
            entries.push((next_id, vec![b1, b2]));
            next_id += 1;
        }
    }

    // A few multi-byte tokens
    for tok in &[
        b"null" as &[u8], b"true", b"fals", b"false", b"image", b"context", b"kaniko",
    ] {
        entries.push((next_id, tok.to_vec()));
        next_id += 1;
    }

    Vocab::new(entries)
}

/// Schema reproducing the o6363 Artifact pattern:
/// `required: ["image"]` at the base, `anyOf` with variants that each
/// have `additionalProperties: false` and their own property sets.
fn artifact_schema() -> String {
    r#"{
        "required": ["image"],
        "anyOf": [
            {
                "additionalProperties": false,
                "properties": {
                    "image": {"type": "string"},
                    "context": {"type": "string"}
                }
            },
            {
                "additionalProperties": false,
                "properties": {
                    "image": {"type": "string"},
                    "kaniko": {"type": "object"}
                }
            }
        ]
    }"#
    .to_string()
}

#[test]
fn test_required_propagated_through_anyof_base_merge() {
    let vocab = build_small_vocab();
    let schema = artifact_schema();
    let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
    let mut state = constraint.start();

    // Valid: `{"image": "x"}` — image is present (required + first in order).
    let valid = b"{\"image\": \"x\"}";
    for &byte in valid.iter() {
        state.commit_bytes(&[byte]).unwrap();
    }
    assert!(state.is_finished());

    // After `{"image": "x", "`, kaniko should be allowed (variant 1
    // allows it after image in declaration order).
    let mut state2 = constraint.start();
    state2.commit_bytes(b"{\"image\": \"x\", \"").unwrap();
    let mask2 = state2.mask();
    let k_id = b'k' as usize;
    assert!(
        mask2[k_id / 32] & (1 << (k_id % 32)) != 0,
        "after image, 'k' (start of 'kaniko') should be allowed in variant 1"
    );

    // After `{"image": "x"}`, closing brace IS allowed (required image present).
    let mut state3 = constraint.start();
    state3.commit_bytes(b"{\"image\": \"x\"").unwrap();
    let mask3 = state3.mask();
    let close_brace_id = b'}' as usize;
    assert!(
        mask3[close_brace_id / 32] & (1 << (close_brace_id % 32)) != 0,
        "closing brace should be allowed when required 'image' is present"
    );

    // After `{"`, `k` (kaniko) should NOT be allowed — image must appear
    // first in ordered grammar since it's declared before kaniko.
    let mut state5 = constraint.start();
    state5.commit_bytes(b"{\"").unwrap();
    let mask5 = state5.mask();
    let k_id = b'k' as usize;
    assert!(
        mask5[k_id / 32] & (1 << (k_id % 32)) == 0,
        "'k' (start of 'kaniko') should be rejected as first property; \
         'image' must come first in ordered grammar"
    );
}
