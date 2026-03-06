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

    let mask = s.compute_mask();
    assert!(mask.get(0), "'a' should be allowed first");
    assert!(!mask.get(1), "'b' should NOT be allowed first");

    s.commit(0);
    let mask = s.compute_mask();
    assert!(!mask.get(0), "'a' should NOT be allowed after 'a'");
    assert!(mask.get(1), "'b' should be allowed after 'a'");

    s.commit(1);
    assert!(s.is_accepting(), "should accept after 'ab'");
}

#[test]
fn test_ebnf_choice() {
    let vocab = make_vocab(&["x", "y", "z"]);
    let c = Constraint::from_ebnf(r#"start ::= "x" | "y""#, &vocab).unwrap();
    let mut s = c.start();

    let mask = s.compute_mask();
    assert!(mask.get(0), "'x' allowed");
    assert!(mask.get(1), "'y' allowed");
    assert!(!mask.get(2), "'z' not allowed");

    s.commit(0);
    assert!(s.is_accepting(), "accept after 'x'");
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

    let mask = s.compute_mask();
    assert!(mask.get(0), "'a' allowed initially");
    assert!(mask.get(1), "'b' allowed initially");
    assert!(!mask.get(2), "'.' not allowed initially");

    s.commit(0); // commit "a"
    let mask = s.compute_mask();
    assert!(mask.get(2), "'.' allowed after 'a'");

    s.commit(2); // commit "."
    assert!(s.is_accepting(), "accept after 'a.'");
}

#[test]
fn test_ebnf_sequence_of_three() {
    let vocab = make_vocab(&["a", "b", "c"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b" "c""#, &vocab).unwrap();
    let mut s = c.start();

    // Step through a → b → c.
    let m = s.compute_mask();
    assert!(m.get(0) && !m.get(1) && !m.get(2));
    s.commit(0);

    let m = s.compute_mask();
    assert!(!m.get(0) && m.get(1) && !m.get(2));
    s.commit(1);

    let m = s.compute_mask();
    assert!(!m.get(0) && !m.get(1) && m.get(2));
    s.commit(2);

    assert!(s.is_accepting());
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

    let mask = s.compute_mask();
    assert!(mask.get(0), "'a' allowed first");
    s.commit(0);

    let mask = s.compute_mask();
    assert!(mask.get(1), "'b' allowed after 'a'");
    s.commit(1);

    assert!(s.is_accepting());
}

#[test]
fn test_lark_choice() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_lark(r#"start: "a" | "b""#, &vocab).unwrap();
    let s = c.start();

    let mask = s.compute_mask();
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
    let mask = s.compute_mask();
    // "t" (token 0) or "f" (token 4) should be allowed.
    assert!(
        mask.get(0) || mask.get(4),
        "boolean start: 't' or 'f' should be allowed"
    );
}

#[test]
fn test_json_schema_null() {
    let vocab = make_vocab(&["n", "u", "l"]);
    let c = Constraint::from_json_schema(r#"{"type": "null"}"#, &vocab).unwrap();
    let mut s = c.start();
    let mask = s.compute_mask();
    assert!(mask.get(0), "'n' allowed for null");

    // Commit "n", "u", "l", "l"
    s.commit(0); // n
    s.commit(1); // u
    s.commit(2); // l
    s.commit(2); // l
    assert!(s.is_accepting(), "accept after 'null'");
}

#[test]
fn test_json_schema_enum() {
    let vocab = make_vocab(&["\"", "a", "b"]);
    let c = Constraint::from_json_schema(r#"{"enum": ["\"a\"", "\"b\""]}"#, &vocab).unwrap();
    let s = c.start();
    let mask = s.compute_mask();
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
    // commit() is infallible: committing a token not in the mask is a no-op.
    // (Token 1 = "b" is NOT allowed by grammar `"a"` at the first step.)
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
    let mut s = c.start();

    // Token 99 is not in the vocabulary at all — should be a no-op.
    s.commit(99);
    let mask = s.compute_mask();
    // State unchanged: "a" still the only valid next token.
    assert!(mask.get(0), "'a' still allowed after no-op commit");
    assert!(!mask.get(1), "'b' still not allowed");
}

#[test]
fn test_multiple_independent_sequences() {
    // Token 0 = "a", Token 1 = "b", Token 2 = "c", Token 3 = "d"
    let vocab = make_vocab(&["a", "b", "c", "d"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b" | "c" "d""#, &vocab).unwrap();
    let mut s = c.start();

    let mask = s.compute_mask();
    assert!(mask.get(0), "'a' allowed initially");
    assert!(mask.get(2), "'c' allowed initially");
    assert!(!mask.get(1), "'b' not allowed initially");
    assert!(!mask.get(3), "'d' not allowed initially");

    // Choose "a" path.
    s.commit(0);
    let mask = s.compute_mask();
    assert!(mask.get(1), "'b' allowed after 'a'");
    assert!(!mask.get(3), "'d' not allowed after 'a'");

    s.commit(1);
    assert!(s.is_accepting(), "accept after 'ab'");
}

// ====================================================================
// Serialization tests
// ====================================================================

#[test]
fn test_save_load_roundtrip() {
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab).unwrap();

    // Serialize and deserialize.
    let bytes = c.save();
    assert!(!bytes.is_empty());
    let c2 = Constraint::load(&bytes).unwrap();

    // The reloaded constraint should behave identically.
    let mut s = c2.start();
    let mask = s.compute_mask();
    assert!(mask.get(0));
    assert!(!mask.get(1));

    s.commit(0);
    let mask = s.compute_mask();
    assert!(mask.get(1));

    s.commit(1);
    assert!(s.is_accepting());
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
    let mask = s.compute_mask();
    assert!(mask.get(0), "'x' allowed");
    assert!(mask.get(2), "'z' allowed");

    // Take the "z" path.
    s.commit(2);
    assert!(s.is_accepting(), "accept after 'z'");

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
fn test_string_backslash_tokens() {
    // Test that backslash tokens that don't form valid escapes are rejected
    let lark = r#"
PATTERN_2: /[\x80-\xBF]/
STRING_CHAR: /[\x20-!#-\x5B\x5D-\x7F]/
    | /[\xC2-\xDF]/ PATTERN_2
    | /[\xE0-\xEF]/ PATTERN_2 PATTERN_2
    | /[\xF0-\xF4]/ PATTERN_2 PATTERN_2 PATTERN_2
HEX: /[0-9A-Fa-f]/
ESCAPE_SHORT_CHAR: /["\x2F\x5Cbfnrt]/
ESCAPE_SEQ: "\\" ESCAPE_SHORT_CHAR | "\\" "u" HEX HEX HEX HEX
STRING_CONTENT: (STRING_CHAR | ESCAPE_SEQ)*
JSON_STRING: "\"" STRING_CONTENT "\""
start: JSON_STRING
"#;
    let vocab = glrmask::Vocab::new(
        vec![
            (1, b"\"".to_vec()),      // opening/closing quote
            (2, b"hello".to_vec()),    // valid string content
            (3, b"\\n".to_vec()),      // valid escape: \n
            (4, b"\\.".to_vec()),      // INVALID escape: \.
            (5, b" \\(".to_vec()),     // space + INVALID escape: \(
            (6, b"\\\"".to_vec()),     // valid escape: \"
        ],
        None,
    );
    let constraint = glrmask::Constraint::from_lark(lark, &vocab).unwrap();

    // Initial state: only " token should be allowed (to start the string)
    let state = constraint.start();
    let mask = state.compute_mask();
    eprintln!("Initial mask: {:?}", (1..=6).filter(|i| mask.get(*i)).collect::<Vec<_>>());
    assert!(mask.get(1), "token 1 (\") should start string");

    // After committing ", we're inside the string
    let mut state = constraint.start();
    state.commit(1); // commit "

    let mask = state.compute_mask();
    let active: Vec<usize> = (1..=6).filter(|i| mask.get(*i)).collect();
    eprintln!("After \": {:?}", active);

    assert!(mask.get(1), "closing quote should be valid");
    assert!(mask.get(2), "hello should be valid string content");
    assert!(mask.get(3), "\\n should be valid escape");
    assert!(!mask.get(4), "\\. should NOT be valid (invalid escape)");
    assert!(!mask.get(5), "space+\\( should NOT be valid (invalid escape)");
    assert!(mask.get(6), "\\\" should be valid escape");
}





#[test]
fn test_array_int_comma_after_digit() {
    // Test that after "[" then "1", the mask includes "," and "]"
    let lark = r#"
DIGIT: /[0-9]/
NONZERO_DIGIT: /[1-9]/
INT_PART: "0" | NONZERO_DIGIT DIGIT*
JSON_INTEGER: "-"? INT_PART
start: "[" "]" | "[" JSON_INTEGER ("," JSON_INTEGER)* "]"
"#;
    // Vocab: 0="[", 1="]", 2=",", 3="1", 4="0", 5="-", 6="23", 7=",-"
    let entries: Vec<(u32, Vec<u8>)> = vec![
        (0, b"[".to_vec()),
        (1, b"]".to_vec()),
        (2, b",".to_vec()),
        (3, b"1".to_vec()),
        (4, b"0".to_vec()),
        (5, b"-".to_vec()),
        (6, b"23".to_vec()),
        (7, b",-".to_vec()),
    ];
    let vocab = Vocab::new(entries, None);
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();

    // Step 0: should allow "[" 
    let mask0 = s.compute_mask();
    eprintln!("Step 0 mask: {:?}", mask0.iter_ones().collect::<Vec<_>>());
    assert!(mask0.get(0), "'[' should be allowed");

    // Commit "["
    s.commit(0);

    // Step 1: should allow digits and "-"
    let mask1 = s.compute_mask();
    eprintln!("Step 1 mask: {:?}", mask1.iter_ones().collect::<Vec<_>>());
    assert!(mask1.get(3), "'1' should be allowed after '['");

    // Commit "1" — trace what the tokenizer does
    eprintln!("\n--- Tracing tokenizer for '1' ---");
    c.debug_tokenizer(b"1", c.tokenizer_initial_state());
    
    // Also trace ",", "]"
    eprintln!("\n--- Tracing tokenizer for ',' ---");
    c.debug_tokenizer(b",", c.tokenizer_initial_state());
    eprintln!("\n--- Tracing tokenizer for ']' ---");
    c.debug_tokenizer(b"]", c.tokenizer_initial_state());
    
    // Debug dump to see terminal IDs and DFA structure
    c.debug_dump();
    
    s.commit(3);

    // Step 2: should allow ",", "]", ",-", and digit tokens
    let mask2 = s.compute_mask();
    let allowed: Vec<usize> = mask2.iter_ones().collect();
    eprintln!("\nStep 2 mask after '[1': {:?}", allowed);
    
    assert!(mask2.get(2), "',' (id=2) should be allowed after '[1'");
    assert!(mask2.get(1), "']' (id=1) should be allowed after '[1'");
    assert!(mask2.get(7), "',-' (id=7) should be allowed after '[1'");

    // Commit ","
    s.commit(2);
    let mask3 = s.compute_mask();
    eprintln!("Step 3 mask after '[1,': {:?}", mask3.iter_ones().collect::<Vec<_>>());
    assert!(mask3.get(3), "'1' should be allowed after ','");

    // Commit "2" (using token "1" which is id=3 with bytes "1")
    // Actually, let's commit token 3 (bytes="1") representing a second digit
    s.commit(3);
    let mask4 = s.compute_mask();
    let allowed4: Vec<usize> = mask4.iter_ones().collect();
    eprintln!("Step 4 mask after '[1,1': {:?}", allowed4);
    assert!(mask4.get(2), "',' (id=2) should be allowed after '[1,1'");
    assert!(mask4.get(1), "']' (id=1) should be allowed after '[1,1'");
    assert!(mask4.get(7), "',-' (id=7) should be allowed after '[1,1'");
}

// ====================================================================
// Plan-conforming API surface tests
// ====================================================================

/// Verify `mask_len()`, `mask()`, `fill_mask()`, and `is_finished()`.
#[test]
fn test_plan_api_mask_and_is_finished() {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec())],
        None,
    );
    let c = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab).unwrap();
    let mut s = c.start();

    // mask_len() must cover every token index.
    let len = c.mask_len();
    assert!(len >= 1, "mask_len must be at least 1");
    assert!((len - 1) * 32 < 32 * len, "mask_len sanity");

    // mask() returns the same information as compute_mask().
    let bitmask = s.compute_mask();
    let words = s.mask();
    assert_eq!(words.len(), len);
    assert!((words[0] >> 0) & 1 == 1, "token 0 ('a') should be set");
    assert!((words[0] >> 1) & 1 == 0, "token 1 ('b') should not be set");

    // fill_mask() must agree with mask().
    let mut buf = vec![0u32; len];
    s.fill_mask(&mut buf);
    assert_eq!(buf, words);

    // is_finished() before completion.
    assert!(!s.is_finished());

    // Advance to completion.
    s.commit(0);
    s.commit(1);
    assert!(s.is_finished());
    let _ = bitmask; // suppress unused warning
}

/// Verify `commit_bytes()` advances state correctly.
#[test]
fn test_plan_api_commit_bytes() {
    let vocab = Vocab::new(
        vec![(0, b"x".to_vec()), (1, b"y".to_vec())],
        None,
    );
    let c = Constraint::from_ebnf(r#"start ::= "x" "y""#, &vocab).unwrap();
    let mut s = c.start();

    // commit_bytes is infallible and processes raw bytes directly.
    s.commit_bytes(b"x");
    let mask = s.mask();
    // After "x", only "y" (token 1) is allowed.
    assert!((mask[0] >> 0) & 1 == 0, "token 0 ('x') must not be set after 'x'");
    assert!((mask[0] >> 1) & 1 == 1, "token 1 ('y') must be set after 'x'");

    s.commit_bytes(b"y");
    assert!(s.is_finished());
}

/// Verify `commit_tokens()` is equivalent to sequential `commit()`.
#[test]
fn test_plan_api_commit_tokens() {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
        None,
    );
    let c = Constraint::from_ebnf(r#"start ::= "a" "b" "c""#, &vocab).unwrap();
    let mut s = c.start();

    s.commit_tokens(&[0, 1, 2]);
    assert!(s.is_finished());
}

/// Verify `force()` returns the forced token sequence for a deterministic grammar.
#[test]
fn test_plan_api_force_deterministic() {
    // Grammar: exactly "a" "b" "c" — all three tokens are forced in one call.
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
        None,
    );
    let c = Constraint::from_ebnf(r#"start ::= "a" "b" "c""#, &vocab).unwrap();
    let s = c.start();

    // force() greedily collects the entire deterministic sequence.
    let forced = s.force();
    assert_eq!(forced, vec![0, 1, 2], "all three tokens are forced in sequence");

    // Committing the forced tokens reaches the finished state.
    let mut s2 = c.start();
    s2.commit_tokens(&forced);
    assert!(s2.is_finished());
}

/// Verify `force()` returns empty when multiple tokens are possible.
#[test]
fn test_plan_api_force_nondeterministic() {
    let vocab = Vocab::new(
        vec![(0, b"x".to_vec()), (1, b"y".to_vec())],
        None,
    );
    let c = Constraint::from_ebnf(r#"start ::= "x" | "y""#, &vocab).unwrap();
    let s = c.start();

    let forced = s.force();
    assert!(forced.is_empty(), "no token forced when two are possible");
}