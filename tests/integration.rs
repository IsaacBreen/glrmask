//! Integration tests: end-to-end from grammar → compile → mask → commit.

use glrmask::{Constraint, Vocab};

/// Build a vocabulary from string entries.
fn make_vocab(entries: &[&str]) -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
        .collect();
    Vocab::new(entries, None)
}

// ====================================================================
// EBNF integration tests
// ====================================================================

#[test]
fn test_ebnf_simple_literal() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab).unwrap();
    let mut s = c.start();

    let mask = s.compute_mask(&c);
    assert!(mask.get(0), "'a' should be allowed first");
    assert!(!mask.get(1), "'b' should NOT be allowed first");

    s.commit(&c, 0).unwrap();
    let mask = s.compute_mask(&c);
    assert!(!mask.get(0), "'a' should NOT be allowed after 'a'");
    assert!(mask.get(1), "'b' should be allowed after 'a'");

    s.commit(&c, 1).unwrap();
    assert!(s.is_accepting(&c), "should accept after 'ab'");
}

#[test]
fn test_ebnf_choice() {
    let vocab = make_vocab(&["x", "y", "z"]);
    let c = Constraint::from_ebnf(r#"start ::= "x" | "y""#, &vocab).unwrap();
    let mut s = c.start();

    let mask = s.compute_mask(&c);
    assert!(mask.get(0), "'x' allowed");
    assert!(mask.get(1), "'y' allowed");
    assert!(!mask.get(2), "'z' not allowed");

    s.commit(&c, 0).unwrap();
    assert!(s.is_accepting(&c), "accept after 'x'");
}

#[test]
fn test_ebnf_multi_rule() {
    let vocab = make_vocab(&["a", "b", "."]);
    let c = Constraint::from_ebnf(
        r#"
        start ::= item "."
        item ::= "a" | "b"
        "#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.compute_mask(&c);
    assert!(mask.get(0), "'a' allowed initially");
    assert!(mask.get(1), "'b' allowed initially");
    assert!(!mask.get(2), "'.' not allowed initially");

    s.commit(&c, 0).unwrap(); // commit "a"
    let mask = s.compute_mask(&c);
    assert!(mask.get(2), "'.' allowed after 'a'");

    s.commit(&c, 2).unwrap(); // commit "."
    assert!(s.is_accepting(&c), "accept after 'a.'");
}

#[test]
fn test_ebnf_sequence_of_three() {
    let vocab = make_vocab(&["a", "b", "c"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b" "c""#, &vocab).unwrap();
    let mut s = c.start();

    // Step through a → b → c.
    let m = s.compute_mask(&c);
    assert!(m.get(0) && !m.get(1) && !m.get(2));
    s.commit(&c, 0).unwrap();

    let m = s.compute_mask(&c);
    assert!(!m.get(0) && m.get(1) && !m.get(2));
    s.commit(&c, 1).unwrap();

    let m = s.compute_mask(&c);
    assert!(!m.get(0) && !m.get(1) && m.get(2));
    s.commit(&c, 2).unwrap();

    assert!(s.is_accepting(&c));
}

// ====================================================================
// Lark integration tests
// ====================================================================

#[test]
fn test_lark_simple() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_lark(
        r#"
        start: "a" "b"
        "#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.compute_mask(&c);
    assert!(mask.get(0), "'a' allowed first");
    s.commit(&c, 0).unwrap();

    let mask = s.compute_mask(&c);
    assert!(mask.get(1), "'b' allowed after 'a'");
    s.commit(&c, 1).unwrap();

    assert!(s.is_accepting(&c));
}

#[test]
fn test_lark_choice() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_lark(r#"start: "a" | "b""#, &vocab).unwrap();
    let mut s = c.start();

    let mask = s.compute_mask(&c);
    assert!(mask.get(0) && mask.get(1));
}

// ====================================================================
// JSON Schema integration tests
// ====================================================================

#[test]
fn test_json_schema_boolean() {
    // Vocabulary: tokens for each character in "true" and "false".
    let vocab = make_vocab(&["t", "r", "u", "e", "f", "a", "l", "s"]);
    let c = Constraint::from_json_schema(r#"{"type": "boolean"}"#, &vocab).unwrap();
    let s = c.start();
    let mask = s.compute_mask(&c);
    // "t" (token 0) or "f" (token 4) should be allowed.
    assert!(mask.get(0) || mask.get(4), "boolean start: 't' or 'f' should be allowed");
}

#[test]
fn test_json_schema_null() {
    let vocab = make_vocab(&["n", "u", "l"]);
    let c = Constraint::from_json_schema(r#"{"type": "null"}"#, &vocab).unwrap();
    let mut s = c.start();
    let mask = s.compute_mask(&c);
    assert!(mask.get(0), "'n' allowed for null");

    // Commit "n", "u", "l", "l"
    s.commit(&c, 0).unwrap(); // n
    s.commit(&c, 1).unwrap(); // u
    s.commit(&c, 2).unwrap(); // l
    s.commit(&c, 2).unwrap(); // l
    assert!(s.is_accepting(&c), "accept after 'null'");
}

#[test]
fn test_json_schema_enum() {
    let vocab = make_vocab(&["\"", "a", "b"]);
    let c = Constraint::from_json_schema(r#"{"enum": ["\"a\"", "\"b\""]}"#, &vocab).unwrap();
    let s = c.start();
    let mask = s.compute_mask(&c);
    // Note: the enum values are JSON strings, so they include quotes.
    // The grammar should start with '"'.
    assert!(mask.get(0), "'\"' allowed for enum start");
}

// ====================================================================
// Error handling tests
// ====================================================================

#[test]
fn test_ebnf_parse_error() {
    let vocab = make_vocab(&["a"]);
    let result = Constraint::from_ebnf("", &vocab);
    assert!(result.is_err());
}

#[test]
fn test_json_schema_invalid_json() {
    let vocab = make_vocab(&["a"]);
    let result = Constraint::from_json_schema("not json", &vocab);
    assert!(result.is_err());
}

// ====================================================================
// Complex grammar tests
// ====================================================================

#[test]
fn test_commit_invalid_token() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
    let mut s = c.start();

    // Token "b" should work but won't lead to accepting.
    // (The commit function may or may not error; depends on runtime impl.)
    let result = s.commit(&c, 1);
    // Even if commit succeeds structurally, the state should be dead.
    if result.is_ok() {
        let mask = s.compute_mask(&c);
        // Should have no valid tokens after an invalid commit.
        assert!(!mask.get(0) && !mask.get(1), "no tokens after invalid commit");
    }
}

#[test]
fn test_multiple_independent_sequences() {
    // Token 0 = "a", Token 1 = "b", Token 2 = "c", Token 3 = "d"
    let vocab = make_vocab(&["a", "b", "c", "d"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b" | "c" "d""#, &vocab).unwrap();
    let mut s = c.start();

    let mask = s.compute_mask(&c);
    assert!(mask.get(0), "'a' allowed initially");
    assert!(mask.get(2), "'c' allowed initially");
    assert!(!mask.get(1), "'b' not allowed initially");
    assert!(!mask.get(3), "'d' not allowed initially");

    // Choose "a" path.
    s.commit(&c, 0).unwrap();
    let mask = s.compute_mask(&c);
    assert!(mask.get(1), "'b' allowed after 'a'");
    assert!(!mask.get(3), "'d' not allowed after 'a'");

    s.commit(&c, 1).unwrap();
    assert!(s.is_accepting(&c), "accept after 'ab'");
}

// ====================================================================
// Serialization tests
// ====================================================================

#[test]
fn test_save_load_roundtrip() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab).unwrap();

    // Serialize and deserialize.
    let bytes = c.save().unwrap();
    assert!(!bytes.is_empty());
    let c2 = Constraint::load(&bytes).unwrap();

    // The reloaded constraint should behave identically.
    let mut s = c2.start();
    let mask = s.compute_mask(&c2);
    assert!(mask.get(0));
    assert!(!mask.get(1));

    s.commit(&c2, 0).unwrap();
    let mask = s.compute_mask(&c2);
    assert!(mask.get(1));

    s.commit(&c2, 1).unwrap();
    assert!(s.is_accepting(&c2));
}

#[test]
fn test_save_load_file_roundtrip() {
    let vocab = make_vocab(&["x", "y", "z"]);
    let c = Constraint::from_ebnf(r#"start ::= "x" "y" | "z""#, &vocab).unwrap();

    let path = std::path::PathBuf::from("/tmp/glrmask_test_roundtrip.bin");
    c.save_to_file(&path).unwrap();
    let c2 = Constraint::load_from_file(&path).unwrap();

    // Verify behavior matches.
    let mut s = c2.start();
    let mask = s.compute_mask(&c2);
    assert!(mask.get(0), "'x' allowed");
    assert!(mask.get(2), "'z' allowed");

    // Take the "z" path.
    s.commit(&c2, 2).unwrap();
    assert!(s.is_accepting(&c2), "accept after 'z'");

    // Cleanup.
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_load_invalid_bytes() {
    let result = Constraint::load(b"not valid bincode");
    assert!(result.is_err());
}

// ====================================================================
// Force token tests
// ====================================================================

#[test]
fn test_forced_token_detection() {
    use glrmask::runtime::force::{forced_token, is_dead};

    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
    let s = c.start();
    let mask = s.compute_mask(&c);

    // Only "a" should be valid — forced token should be 0.
    assert_eq!(forced_token(&mask), Some(0));
    assert!(!is_dead(&mask));
}
