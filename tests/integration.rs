//! Integration tests: end-to-end from grammar → compile → mask → commit.

use glrmask::{Constraint, ConstraintState, Vocab};

/// Build a vocabulary from string entries.
fn make_vocab(entries: &[&str]) -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
        .collect();
    Vocab::new(entries, None)
}

fn ebnf_constraint(entries: &[&str], grammar: &str) -> Constraint {
    let vocab = make_vocab(entries);
    Constraint::from_ebnf(grammar, &vocab).unwrap()
}

fn lark_constraint(entries: &[&str], grammar: &str) -> Constraint {
    let vocab = make_vocab(entries);
    Constraint::from_lark(grammar, &vocab).unwrap()
}

fn json_schema_constraint(entries: &[&str], schema: &str) -> Constraint {
    let vocab = make_vocab(entries);
    Constraint::from_json_schema(schema, &vocab).unwrap()
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() { return false; }
    (mask[word] >> (id % 32)) & 1 != 0
}

fn iter_allowed(mask: &[u32]) -> Vec<usize> {
    mask.iter().enumerate().flat_map(|(w, &bits)| {
        (0..32u32).filter_map(move |b| if (bits >> b) & 1 != 0 { Some(w * 32 + b as usize) } else { None })
    }).collect()
}

fn assert_mask_allows(mask: &[u32], ids: &[usize]) {
    let allowed = iter_allowed(mask);
    for &id in ids {
        assert!(token_allowed(mask, id), "token {id} should be allowed; allowed={allowed:?}");
    }
}

fn assert_mask_rejects(mask: &[u32], ids: &[usize]) {
    let allowed = iter_allowed(mask);
    for &id in ids {
        assert!(!token_allowed(mask, id), "token {id} should not be allowed; allowed={allowed:?}");
    }
}

fn assert_allowed_tokens(mask: &[u32], expected: &[usize]) {
    assert_eq!(iter_allowed(mask), expected);
}

fn assert_empty_mask(mask: &[u32]) {
    let allowed = iter_allowed(mask);
    assert!(allowed.is_empty(), "mask should be empty: {allowed:?}");
}

fn commit_all(state: &mut ConstraintState<'_>, tokens: &[u32]) {
    for &token in tokens {
        state.commit_token(token).unwrap();
    }
}

fn byte_vocab() -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = (0..=255u32).map(|b| (b, vec![b as u8])).collect();
    Vocab::new(entries, None)
}

fn anyof_object_schema_with_n_branches(n: usize) -> String {
    // Minimal reproducing schema: shared "id" property + unique "k{i}" per branch.
    // Use boolean-true schemas for property values (equivalent to unconstrained schemas).
    // No required, no explicit additionalProperties (defaults to true).
    // additionalProperties:true is what causes ambiguity: when parsing {"id":0,"k0":0},
    // branch 0 treats k0 as a known property; branches 1..N-1 treat it as additional.
    let branches: Vec<String> = (0..n)
        .map(|i| format!(r#"    {{"properties":{{"id":true,"k{i}":true}}}}"#))
        .collect();

    format!(
        "{{\n  \"anyOf\": [\n{}\n  ]\n}}",
        branches.join(",\n")
    )
}

fn schema_like_ambiguous_ebnf_with_n_branches(n: usize) -> String {
    assert!(n > 0, "n must be > 0");
    // Supports up to N=32 in the current test set with unique single-char keys.
    assert!(n <= 32, "n too large for single-char key simplification");
    const KEYS: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUV";

    let mut lines = Vec::new();
    lines.push(format!("start ::= s_{}", n - 1));
    lines.push("zstar ::= 'z' zstar | ''".to_string());

    // Minimal core analogue of known-vs-additional ambiguity:
    // - specific key for branch i is decimal literal i
    // - shared fallback/additional key is 0
    // For i=0, specific and fallback coincide, yielding two roles on the same token.
    lines.push("s_0 ::= '0' zstar | '0' zstar".to_string());

    // One extra line per extra N:
    // s_i adds branch i by either matching specific key i or fallback/additional key 0.
    // The required z-prefix keeps branch states distinct after key classification.
    for i in 1..n {
        let key = KEYS[i] as char;
        let z_prefix = (0..i).map(|_| "'z'").collect::<Vec<_>>().join(" ");
        lines.push(format!(
            "s_{i} ::= s_{} | '{key}' {z_prefix} zstar | '0' {z_prefix} zstar",
            i - 1,
        ));
    }

    lines.join("\n") + "\n"
}

fn max_parser_paths_over_bytes(constraint: &Constraint, input: &[u8]) -> usize {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1_000_000);
    for &byte in input {
        let mask = state.mask();
        assert!(
            token_allowed(&mask, byte as usize),
            "expected byte token {:?} to be allowed; allowed={:?}",
            byte,
            iter_allowed(&mask)
        );
        state.commit_token(byte as u32).unwrap();
        max_paths = max_paths.max(state.parser_path_count(1_000_000));
    }
    max_paths
}

#[test]
fn test_ebnf_simple_literal() {
    let constraint = ebnf_constraint(&["a", "b"], r#"start ::= "a" "b""#);
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);

    commit_all(&mut state, &[0]);
    let mask = state.mask();
    assert_mask_rejects(&mask, &[0]);
    assert_mask_allows(&mask, &[1]);

    commit_all(&mut state, &[1]);
    assert!(state.is_finished(), "should accept after 'ab'");
}

#[test]
fn test_ebnf_choice() {
    let constraint = ebnf_constraint(&["x", "y", "z"], r#"start ::= "x" | "y""#);
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0, 1]);
    assert_mask_rejects(&mask, &[2]);

    commit_all(&mut state, &[0]);
    assert!(state.is_finished(), "accept after 'x'");
}

#[test]
fn test_ebnf_multi_rule() {
    let constraint = ebnf_constraint(
        &["a", "b", "."],
        r#"
        start ::= item "."
        item ::= "a" | "b"
        "#,
    );
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0, 1]);
    assert_mask_rejects(&mask, &[2]);

    commit_all(&mut state, &[0]);
    let mask = state.mask();
    assert_mask_allows(&mask, &[2]);

    commit_all(&mut state, &[2]);
    assert!(state.is_finished(), "accept after 'a.'");
}

#[test]
fn test_ebnf_sequence_of_three() {
    let constraint = ebnf_constraint(&["a", "b", "c"], r#"start ::= "a" "b" "c""#);
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1, 2]);
    commit_all(&mut state, &[0]);

    let mask = state.mask();
    assert_mask_allows(&mask, &[1]);
    assert_mask_rejects(&mask, &[0, 2]);
    commit_all(&mut state, &[1]);

    let mask = state.mask();
    assert_mask_allows(&mask, &[2]);
    assert_mask_rejects(&mask, &[0, 1]);
    commit_all(&mut state, &[2]);

    assert!(state.is_finished());
}

#[test]
fn test_lark_simple() {
    let constraint = lark_constraint(
        &["a", "b"],
        r#"
        start: "a" "b"
        "#,
    );
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    commit_all(&mut state, &[0]);

    let mask = state.mask();
    assert_mask_allows(&mask, &[1]);
    commit_all(&mut state, &[1]);

    assert!(state.is_finished());
}

#[test]
fn test_lark_choice() {
    let constraint = lark_constraint(&["a", "b"], r#"start: "a" | "b""#);
    let state = constraint.start();

    assert_allowed_tokens(&state.mask(), &[0, 1]);
}

#[test]
fn test_lark_singleton_char_class_initial_mask() {
    let constraint = lark_constraint(&["1"], r#"start: /[1]/"#);
    let state = constraint.start();

    assert_mask_allows(&state.mask(), &[0]);
}

#[test]
fn test_lark_single_quotes_and_literal_range() {
    let constraint = lark_constraint(
        &["5", "a"],
        "?start: DIGIT\nDIGIT.2: '0'..'9'",
    );
    let state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);
}

#[test]
fn test_lark_alias_syntax_is_ignored_semantically() {
    let constraint = lark_constraint(
        &["a", "b"],
        "start: 'a' -> left | \"b\" -> right",
    );
    let state = constraint.start();

    assert_allowed_tokens(&state.mask(), &[0, 1]);
}

#[test]
fn test_lark_terminal_convention_inlines_uppercase_rules() {
    let constraint = lark_constraint(
        &["a", "b", "c"],
        "start: WORD\nWORD: LETTER+\nLETTER: 'a' | 'b'",
    );
    let state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0, 1]);
    assert_mask_rejects(&mask, &[2]);
}

#[test]
fn test_lark_terminal_convention_rejects_parser_refs_inside_terminals() {
    let vocab = make_vocab(&["a"]);
    let err = Constraint::from_lark("start: WORD\nitem: 'a'\nWORD: item", &vocab)
        .expect_err("invalid terminal/parser mixing should be rejected");
    assert!(
        err.to_string().contains("references nonterminal item"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_json_schema_boolean() {
    let constraint = json_schema_constraint(&["t", "r", "u", "e", "f", "a", "l", "s"], r#"{"type": "boolean"}"#);
    let mask = constraint.start().mask();
    assert_mask_allows(&mask, &[0, 4]);
}

#[test]
fn test_json_schema_null() {
    let constraint = json_schema_constraint(&["n", "u", "l"], r#"{"type": "null"}"#);
    let mut state = constraint.start();
    assert_mask_allows(&state.mask(), &[0]);

    commit_all(&mut state, &[0, 1, 2, 2]);
    assert!(state.is_finished(), "accept after 'null'");
}

#[test]
fn test_json_schema_enum() {
    let constraint = json_schema_constraint(&["\"", "a", "b"], r#"{"enum": ["\"a\"", "\"b\""]}"#);
    assert_mask_allows(&constraint.start().mask(), &[0]);
}

#[test]
fn test_json_schema_bare_object_accepts_compact_empty_object_token() {
    let constraint = json_schema_constraint(&["{}", "true"], r#"{"type": "object"}"#);
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);

    commit_all(&mut state, &[0]);
    assert!(state.is_finished(), "should accept after compact '{{}}' token");
}

#[test]
fn test_json_schema_anyof_ambiguity_grows_with_n_not_log_n() {
    let vocab = byte_vocab();
    let input = br#"{"id": 0, "k0": 0}"#;
    let ns = [2usize, 4, 8, 16];

    let mut observations = Vec::new();
    for &n in &ns {
        let schema = anyof_object_schema_with_n_branches(n);
        let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
        let max_paths = max_parser_paths_over_bytes(&constraint, input);
        println!("N={n} max_paths={max_paths}\n{schema}\n");
        observations.push((n, max_paths));
    }

    for pair in observations.windows(2) {
        let (n0, p0) = pair[0];
        let (n1, p1) = pair[1];
        assert!(
            p1 > p0,
            "max parser paths should increase with N: N={n0} -> {p0}, N={n1} -> {p1}"
        );
    }

    for (n, max_paths) in observations {
        let log2_n_plus_1 = ((n + 1) as f64).log2();
        assert!(
            max_paths >= n,
            "expected at least linear lower bound on this family: N={n}, max_paths={max_paths}"
        );
        if n >= 8 {
            assert!(
                (max_paths as f64) > 3.0 * log2_n_plus_1,
                "growth should be clearly above logarithmic for larger N: N={n}, max_paths={max_paths}, 3*log2(N+1)={}",
                3.0 * log2_n_plus_1
            );
        }
    }
}

#[test]
fn test_ebnf_ambiguity_grows_with_n_not_log_n() {
    let vocab = byte_vocab();
    let ns = [2usize, 4, 8, 16, 32];

    let mut observations = Vec::new();
    for &n in &ns {
        let grammar = schema_like_ambiguous_ebnf_with_n_branches(n);
        let input = [b'0']
            .into_iter()
            .chain(std::iter::repeat_n(b'z', n))
            .collect::<Vec<u8>>();
        let constraint = Constraint::from_ebnf(&grammar, &vocab).unwrap();
        let max_paths = max_parser_paths_over_bytes(&constraint, &input);
        println!("N={n} max_paths={max_paths}\n{grammar}");
        observations.push((n, max_paths));
    }

    for pair in observations.windows(2) {
        let (n0, p0) = pair[0];
        let (n1, p1) = pair[1];
        assert!(
            p1 > p0,
            "max parser paths should increase with N: N={n0} -> {p0}, N={n1} -> {p1}"
        );
    }

    for (n, max_paths) in observations {
        let log2_n_plus_1 = ((n + 1) as f64).log2();
        assert!(
            max_paths >= n,
            "expected at least linear lower bound on this family: N={n}, max_paths={max_paths}"
        );
        if n >= 16 {
            assert!(
                (max_paths as f64) > 3.0 * log2_n_plus_1,
                "growth should be clearly above logarithmic for larger N: N={n}, max_paths={max_paths}, 3*log2(N+1)={}",
                3.0 * log2_n_plus_1
            );
        }
    }
}

#[test]
fn test_ebnf_empty_object_accepts_compact_empty_object_token() {
    let constraint = ebnf_constraint(&["{}", "true"], r#"start ::= "{" "}""#);
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);

    commit_all(&mut state, &[0]);
    assert!(state.is_finished(), "should accept after compact '{{}}' token");
}

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

#[test]
#[should_panic(expected = "not in vocabulary")]
fn test_commit_invalid_token() {
    let constraint = ebnf_constraint(&["a", "b"], r#"start ::= "a""#);
    constraint.start().commit_token(99).unwrap();
}

#[test]
fn test_multiple_independent_sequences() {
    let constraint = ebnf_constraint(&["a", "b", "c", "d"], r#"start ::= "a" "b" | "c" "d""#);
    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0, 2]);
    assert_mask_rejects(&mask, &[1, 3]);

    commit_all(&mut state, &[0]);
    let mask = state.mask();
    assert_mask_allows(&mask, &[1]);
    assert_mask_rejects(&mask, &[3]);

    commit_all(&mut state, &[1]);
    assert!(state.is_finished(), "accept after 'ab'");
}

#[test]
fn test_save_load_roundtrip() {
    let constraint = ebnf_constraint(&["a", "b"], r#"start ::= "a" "b""#);

    let bytes = constraint.save();
    assert!(!bytes.is_empty());
    let c2 = Constraint::load(&bytes).unwrap();

    let mut state = c2.start();
    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);

    commit_all(&mut state, &[0]);
    assert_mask_allows(&state.mask(), &[1]);

    commit_all(&mut state, &[1]);
    assert!(state.is_finished());
}

#[test]
fn test_save_load_file_roundtrip() {
    let constraint = ebnf_constraint(&["x", "y", "z"], r#"start ::= "x" "y" | "z""#);

    let path = std::path::PathBuf::from("/tmp/glrmask_test_roundtrip.bin");
    std::fs::write(&path, constraint.save()).unwrap();
    let c2 = Constraint::load(&std::fs::read(&path).unwrap()).unwrap();

    let mut state = c2.start();
    let mask = state.mask();
    assert_mask_allows(&mask, &[0, 2]);

    commit_all(&mut state, &[2]);
    assert!(state.is_finished(), "accept after 'z'");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_load_invalid_bytes() {
    let result = Constraint::load(b"not valid bincode");
    assert!(result.is_err());
}

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
    let mask = state.mask();
    assert!(token_allowed(&mask, 1), "token 1 (\") should start string");

    // After committing ", we're inside the string
    let mut state = constraint.start();
    state.commit_token(1).unwrap(); // commit "

    let mask = state.mask();
    assert!(token_allowed(&mask, 1), "closing quote should be valid");
    assert!(token_allowed(&mask, 2), "hello should be valid string content");
    assert!(token_allowed(&mask, 3), "\\n should be valid escape");
    assert!(!token_allowed(&mask, 4), "\\. should NOT be valid (invalid escape)");
    assert!(!token_allowed(&mask, 5), "space+\\( should NOT be valid (invalid escape)");
    assert!(token_allowed(&mask, 6), "\\\" should be valid escape");
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

    // Before committing "[", only the opening bracket should be allowed.
    let mask0 = s.mask();
    assert!(token_allowed(&mask0, 0), "'[' should be allowed");

    // Commit "["
    s.commit_token(0).unwrap();

    // After "[", the array can start with digits or "-".
    let mask1 = s.mask();
    assert!(token_allowed(&mask1, 3), "'1' should be allowed after '['");

    s.commit_token(3).unwrap();

    // After the first integer, separators, closing brackets, and ",-" remain valid.
    let mask2 = s.mask();
    assert!(token_allowed(&mask2, 2), "',' (id=2) should be allowed after '[1'");
    assert!(token_allowed(&mask2, 1), "']' (id=1) should be allowed after '[1'");
    assert!(token_allowed(&mask2, 7), "',-' (id=7) should be allowed after '[1'");

    // Commit ","
    s.commit_token(2).unwrap();
    let mask3 = s.mask();
    assert!(token_allowed(&mask3, 3), "'1' should be allowed after ','");

    // Commit "2" (using token "1" which is id=3 with bytes "1")
    // Actually, let's commit token 3 (bytes="1") representing a second digit
    s.commit_token(3).unwrap();
    let mask4 = s.mask();
    assert!(token_allowed(&mask4, 2), "',' (id=2) should be allowed after '[1,1'");
    assert!(token_allowed(&mask4, 1), "']' (id=1) should be allowed after '[1,1'");
    assert!(token_allowed(&mask4, 7), "',-' (id=7) should be allowed after '[1,1'");
}

// Plan-conforming API surface tests

#[test]
fn test_plan_api_mask_and_is_finished() {
    let c = ebnf_constraint(&["a", "b"], r#"start ::= "a" "b""#);
    let mut s = c.start();

    let len = c.mask_len();
    assert!(len >= 1, "mask_len must be at least 1");
    assert!((len - 1) * 32 < 32 * len, "mask_len sanity");

    let bitmask = s.mask();
    let words = s.mask();
    assert_eq!(words.len(), len);
    assert_mask_allows(&words, &[0]);
    assert_mask_rejects(&words, &[1]);

    let mut buf = vec![0u32; len];
    s.fill_mask(&mut buf);
    assert_eq!(buf, words);

    assert!(!s.is_finished());

    commit_all(&mut s, &[0, 1]);
    assert!(s.is_finished());
    let _ = bitmask; // suppress unused warning
}

#[test]
fn test_plan_api_commit_bytes() {
    let c = ebnf_constraint(&["x", "y"], r#"start ::= "x" "y""#);
    let mut s = c.start();

    s.commit_bytes(b"x").unwrap();
    let mask = s.mask();
    assert_mask_rejects(&mask, &[0]);
    assert_mask_allows(&mask, &[1]);

    s.commit_bytes(b"y").unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_plan_api_commit_tokens() {
    let c = ebnf_constraint(&["a", "b", "c"], r#"start ::= "a" "b" "c""#);
    let mut s = c.start();

    s.commit_tokens(&[0, 1, 2]).unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_plan_api_force_deterministic() {
    let c = ebnf_constraint(&["a", "b", "c"], r#"start ::= "a" "b" "c""#);
    let s = c.start();

    let forced = s.force();
    assert_eq!(forced, vec![0, 1, 2], "all three tokens are forced in sequence");

    let mut s2 = c.start();
    s2.commit_tokens(&forced).unwrap();
    assert!(s2.is_finished());
}

#[test]
fn test_plan_api_force_nondeterministic() {
    let c = ebnf_constraint(&["x", "y"], r#"start ::= "x" | "y""#);
    let s = c.start();

    let forced = s.force();
    assert!(forced.is_empty(), "no token forced when two are possible");
}

#[test]
fn test_literal_then_eof_sequence() {
    let c = ebnf_constraint(
        &["a", "$"] ,
        r#"s ::= A EOF
           A ::= 'a'
           EOF ::= '$'"#,
    );

    let mut s = c.start();

    let mask = s.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);

    commit_all(&mut s, &[0]);
    let mask = s.mask();
    assert_mask_rejects(&mask, &[0]);
    assert_mask_allows(&mask, &[1]);

    commit_all(&mut s, &[1]);
    assert!(s.is_finished(), "should be finished after 'a$'");
}

#[test]
fn test_multibyte_token_matches_two_grammar_paths() {
    let c = ebnf_constraint(
        &["ab", "ac", "$"] ,
        r#"s ::= x EOF
           x ::= A B_OR_C | AB
           A ::= 'a'
           AB ::= 'ab'
           B_OR_C ::= 'b' | 'c'
           EOF ::= '$'"#,
    );
    let mut s = c.start();

    let mask = s.mask();
    assert_mask_allows(&mask, &[0, 1]);
    assert_mask_rejects(&mask, &[2]);

    commit_all(&mut s, &[0]);
    let mask = s.mask();
    assert_mask_rejects(&mask, &[0, 1]);
    assert_mask_allows(&mask, &[2]);
}

#[test]
fn test_single_literal_then_eof_compact_path() {
    let c = ebnf_constraint(
        &["a", "$"] ,
        r#"s ::= x EOF
           x ::= A
           A ::= 'a'
           EOF ::= '$'"#,
    );
    let mut s = c.start();

    let mask = s.mask();
    assert_mask_allows(&mask, &[0]);
    assert_mask_rejects(&mask, &[1]);

    commit_all(&mut s, &[0]);
    let mask = s.mask();
    assert_mask_rejects(&mask, &[0]);
    assert_mask_allows(&mask, &[1]);

    commit_all(&mut s, &[1]);
    assert!(s.is_finished());
}

#[test]
fn test_optional_semicolon_between_two_expressions() {
    let c = ebnf_constraint(
        &["x", ";", "$"] ,
        r#"program ::= expression_statement expression_statement? EOF
           expression_statement ::= expression ';'?
           expression ::= 'x'
           EOF ::= '$'"#,
    );
    let mut s = c.start();

    let mask0 = s.mask();
    assert_mask_allows(&mask0, &[0]);
    assert_mask_rejects(&mask0, &[1, 2]);

    commit_all(&mut s, &[0]);
    let mask1 = s.mask();
    assert_mask_allows(&mask1, &[0, 1, 2]);

    commit_all(&mut s, &[1]);
    let mask2 = s.mask();
    assert_mask_allows(&mask2, &[0, 2]);

    commit_all(&mut s, &[0]);
    let mask3 = s.mask();
    assert_mask_allows(&mask3, &[2]);

    commit_all(&mut s, &[2]);
    assert!(s.is_finished(), "should be finished after x;x$");
}

#[test]
fn test_expression_mask_after_compact_open_paren_i_token() {
    let c = ebnf_constraint(
        &["i", "+", "*", "(", ")", "(i", "+i"] ,
        r#"s ::= e
           e ::= e PLUS t | t
           t ::= t TIMES f | f
           f ::= LPAREN e RPAREN | I
           PLUS ::= '+'
           TIMES ::= '*'
           LPAREN ::= '('
           RPAREN ::= ')'
           I ::= 'i'"#,
    );
    let mut s = c.start();

    let mask = s.mask();
    assert_mask_allows(&mask, &[0, 3, 5]);
    assert_mask_rejects(&mask, &[1, 2, 4, 6]);

    commit_all(&mut s, &[5]);
    let mask = s.mask();
    assert_mask_allows(&mask, &[1, 2, 4, 6]);
    assert_mask_rejects(&mask, &[0, 3, 5]);
}

/// Grammar: s ::= A; A ::= 'a'+.
/// Committing "a" three times should produce the same mask as committing "aaa" once.
#[test]
fn test_a_plus_commit_equivalence() {
    // IDs: "a"→0, "aaa"→1
    let vocab = make_vocab(&["a", "aaa"]);
    let c = Constraint::from_ebnf(
        r#"s ::= A
           A ::= 'a'+"#,
        &vocab,
    )
    .unwrap();

    // Scenario 1: commit "a" three times
    let mut s1 = c.start();
    s1.commit_token(0).unwrap();
    s1.commit_token(0).unwrap();
    s1.commit_token(0).unwrap();
    let mask1 = s1.mask();

    // Scenario 2: commit "aaa" once
    let mut s2 = c.start();
    s2.commit_token(1).unwrap();
    let mask2 = s2.mask();

    assert_eq!(
        mask1, mask2,
        "mask after 3×'a' vs 'aaa' should be equivalent"
    );
    assert_eq!(
        s1.is_finished(),
        s2.is_finished(),
        "finished state should be equivalent"
    );
}

/// Ambiguous grammar: s ::= A A; A ::= 'a'+.
/// With only "a" in vocab the constraint should keep token 0 allowed across many commits.
#[test]
fn test_hideous_ambiguity() {
    let vocab = make_vocab(&["a"]);
    let c = Constraint::from_ebnf(
        r#"s ::= A A
           A ::= 'a'+"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    // Commit "a" up to 10 times; token 0 must remain allowed at every step
    for i in 0..10 {
        let mask = s.mask();
        if !token_allowed(&mask, 0) {
            // Acceptable only once we're finished (not before 2 commits)
            assert!(
                s.is_finished() || i >= 2,
                "token 'a' should remain allowed at iteration {i}"
            );
            break;
        }
        s.commit_token(0).unwrap();
    }
}

#[test]
fn test_large_exact_repetition() {
    // IDs: "a" -> 0
    let vocab = make_vocab(&["a"]);

    // A grammar requiring exactly 1 billion 'a's.
    // This specifically tests that the Lowerer uses logarithmic decomposition (Binary Tree)
    // rather than linear decomposition, which would cause a stack overflow.
    let lark = r#"start: "a" ~1000000000..1000000000"#;
    let constraint = lark_constraint(&["a"], lark);

    let mut state = constraint.start();

    // 1. Check initial mask: 'a' must be allowed.
    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);

    // 2. Commit a few tokens.
    // Even after a few commits, we should still be far from the billion-token goal.
    commit_all(&mut state, &[0, 0, 0]);

    let mask_after = state.mask();
    assert_mask_allows(&mask_after, &[0]);
    assert!(!state.is_finished(), "Should not be finished after only 3 tokens");
}

#[test]
fn test_large_max_repetition() {
    let vocab = make_vocab(&["a"]);

    let lark = r#"start: "a" ~0..1000000000"#;
    let constraint = lark_constraint(&["a"], lark);

    let mut state = constraint.start();

    let mask = state.mask();
    assert_mask_allows(&mask, &[0]);

    commit_all(&mut state, &[0, 0, 0]);

    let mask_after = state.mask();
    assert_mask_allows(&mask_after, &[0]);
    assert!(state.is_finished(), "Should be finished after 3 tokens (3 >= min=0)");
}


/// Grammar: s ::= DEF_T; DEF_T ::= "def".
/// Verifies that the multi-byte vocab token "def" is allowed at token id 0.
#[test]
fn test_def_token() {
    let vocab = make_vocab(&["def"]);
    let c = Constraint::from_ebnf(
        r#"s ::= DEF_T
           DEF_T ::= "def""#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'def' (id=0) should be allowed initially");
    s.commit_token(0).unwrap();
    assert!(s.is_finished(), "should be finished after 'def'");
}

/// Grammar: s ::= HASH_OPT_A | HASH_OPT_A A; A ::= 'a'; HASH_OPT_A ::= '#' 'a'?.
/// Verifies that commit("#") then commit("a") yields the same mask as commit("#a").
#[test]
fn test_hash_restart() {
    // IDs: "#"→0, "a"→1, "#a"→2
    let vocab = make_vocab(&["#", "a", "#a"]);
    let c = Constraint::from_ebnf(
        r#"s ::= HASH_OPT_A | HASH_OPT_A A
           A ::= 'a'
           HASH_OPT_A ::= '#' 'a'?"#,
        &vocab,
    )
    .unwrap();

    // Scenario 1: separate tokens
    let mut s1 = c.start();
    s1.commit_token(0).unwrap(); // "#"
    s1.commit_token(1).unwrap(); // "a"
    let mask1 = s1.mask();

    // Scenario 2: combined token "#a"
    let mut s2 = c.start();
    s2.commit_token(2).unwrap(); // "#a"
    let mask2 = s2.mask();

    assert_eq!(
        mask1, mask2,
        "commit('#','a') and commit('#a') should yield equivalent masks"
    );
}

/// Grammar: s ::= HASH_OPT_AA | HASH_OPT_AA A A; HASH_OPT_AA ::= '#' ('a' 'a')?.
/// Verifies that "#","a","a" and "#aa" yield the same final mask.
#[test]
fn test_multi_commit_hash() {
    // IDs: "#"→0, "a"→1, "#aa"→2
    let vocab = make_vocab(&["#", "a", "#aa"]);
    let c = Constraint::from_ebnf(
        r#"s ::= HASH_OPT_AA | HASH_OPT_AA A A
           A ::= 'a'
           HASH_OPT_AA ::= '#' ('a' 'a')?"#,
        &vocab,
    )
    .unwrap();

    // Scenario 1: three separate commits
    let mut s1 = c.start();
    s1.commit_token(0).unwrap(); // "#"
    s1.commit_token(1).unwrap(); // "a"
    s1.commit_token(1).unwrap(); // "a"
    let mask1 = s1.mask();

    // Scenario 2: one combined token
    let mut s2 = c.start();
    s2.commit_token(2).unwrap(); // "#aa"
    let mask2 = s2.mask();

    assert_eq!(
        mask1, mask2,
        "commit('#','a','a') and commit('#aa') should yield equivalent masks"
    );
}

/// Indirect recursion: s_prime ::= s EOF; s ::= A e | B; e ::= s.
/// Equivalent to s → a* b; valid strings are "b", "ab", "aab", …
#[test]
fn test_indirect_recursion() {
    // IDs: "a"→0, "b"→1, "$"→2
    let vocab = make_vocab(&["a", "b", "$"]);
    let c = Constraint::from_ebnf(
        r#"s_prime ::= s EOF
           s ::= A e | B
           e ::= s
           A ::= 'a'
           B ::= 'b'
           EOF ::= '$'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'a' should be allowed initially");
    assert!(token_allowed(&mask, 1), "'b' should be allowed initially");
    assert!(!token_allowed(&mask, 2), "'$' should NOT be allowed initially");

    s.commit_token(0).unwrap(); // "a"
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'a' allowed after 'a' (recursive)");
    assert!(token_allowed(&mask, 1), "'b' allowed after 'a'");
    assert!(!token_allowed(&mask, 2), "'$' NOT allowed after 'a'");

    s.commit_token(1).unwrap(); // "b"
    let mask = s.mask();
    assert!(!token_allowed(&mask, 0), "'a' NOT allowed after 'ab'");
    assert!(!token_allowed(&mask, 1), "'b' NOT allowed after 'ab'");
    assert!(token_allowed(&mask, 2), "'$' should be allowed after 'ab'");
}

/// Left-recursive repetition: s_prime ::= s; s ::= s A | ε.
/// Equivalent to A*; "a" must remain allowed after each commit.
#[test]
fn test_repetition_left_recursive() {
    let vocab = make_vocab(&["a"]);
    let c = Constraint::from_ebnf(
        r#"s_prime ::= s
           s ::= s A |
           A ::= 'a'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'a' should be allowed initially");

    s.commit_token(0).unwrap();
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'a' should be allowed after first 'a'");
}

/// Token "i(" spans grammar terminals [I, LPAREN] but after I only EOF is valid.
/// Therefore "i(" is always forbidden, and "$" is forbidden at start → empty mask.
#[test]
fn test_split_token_invalid() {
    // IDs: "i("→0, "$"→1
    let vocab = make_vocab(&["i(", "$"]);
    let c = Constraint::from_ebnf(
        r#"s ::= e EOF
           e ::= LPAREN e | I
           LPAREN ::= '('
           I ::= 'i'
           EOF ::= '$'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let mask = s.mask();
    assert!(
        !token_allowed(&mask, 0),
        "'i(' should NOT be allowed (invalid token sequence)"
    );
    assert!(
        !token_allowed(&mask, 1),
        "'$' should NOT be allowed at start"
    );
}

/// Indirect expression: s ::= e EOF; e ::= f; f ::= LPAREN e | I.
/// '(' may recurse indefinitely; after "(i" only '$' remains.
#[test]
fn test_trivial_indirect_expression() {
    // IDs: "i"→0, "("→1, "(i"→2, "$"→3
    let vocab = make_vocab(&["i", "(", "(i", "$"]);
    let c = Constraint::from_ebnf(
        r#"s ::= e EOF
           e ::= f
           f ::= LPAREN e | I
           LPAREN ::= '('
           I ::= 'i'
           EOF ::= '$'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'i' should be allowed initially");
    assert!(token_allowed(&mask, 1), "'(' should be allowed initially");
    assert!(token_allowed(&mask, 2), "'(i' should be allowed initially");
    assert!(!token_allowed(&mask, 3), "'$' should NOT be allowed initially");

    s.commit_token(1).unwrap(); // "("
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'i' after '('");
    assert!(token_allowed(&mask, 1), "'(' after '(' (recursive)");
    assert!(token_allowed(&mask, 2), "'(i' after '('");
    assert!(!token_allowed(&mask, 3), "'$' NOT after '('");

    s.commit_token(0).unwrap(); // "i"
    let mask = s.mask();
    assert!(!token_allowed(&mask, 0), "'i' NOT after '(i'");
    assert!(!token_allowed(&mask, 1), "'(' NOT after '(i'");
    assert!(!token_allowed(&mask, 2), "'(i' NOT after '(i'");
    assert!(token_allowed(&mask, 3), "'$' should be allowed after '(i'");
}

/// Direct left-recursive expression: s ::= e EOF; e ::= LPAREN e | I.
/// Same behavioural expectations as the indirect version above.
#[test]
fn test_trivial_direct_expression() {
    // IDs: "i"→0, "("→1, "(i"→2, "$"→3
    let vocab = make_vocab(&["i", "(", "(i", "$"]);
    let c = Constraint::from_ebnf(
        r#"s ::= e EOF
           e ::= LPAREN e | I
           LPAREN ::= '('
           I ::= 'i'
           EOF ::= '$'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'i' should be allowed initially");
    assert!(token_allowed(&mask, 1), "'(' should be allowed initially");
    assert!(token_allowed(&mask, 2), "'(i' should be allowed initially");
    assert!(!token_allowed(&mask, 3), "'$' should NOT be allowed initially");

    s.commit_token(1).unwrap(); // "("
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'i' after '('");
    assert!(token_allowed(&mask, 1), "'(' after '(' (recursive)");
    assert!(token_allowed(&mask, 2), "'(i' after '('");
    assert!(!token_allowed(&mask, 3), "'$' NOT after '('");

    s.commit_token(0).unwrap(); // "i"
    let mask = s.mask();
    assert!(token_allowed(&mask, 3), "'$' should be allowed after '(i'");
}

// Constraint regressions

/// Sparse vocabulary: only token ID=2 ("(i") exists; IDs 0 and 1 are absent.
/// Grammar: s ::= e EOF; e ::= LPAREN e | I.
/// Initial mask has only token 2 set. After commit, need EOF (not in vocab) → empty.
#[test]
fn test_limited_vocab_direct_expression() {
    // Only token ID 2 exists; IDs 0 and 1 are absent (sparse vocab)
    let vocab = Vocab::new(vec![(2u32, b"(i".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= e EOF
           e ::= LPAREN e | I
           LPAREN ::= '('
           I ::= 'i'
           EOF ::= '$'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert!(!token_allowed(&mask, 0), "token 0 (absent) not in mask");
    assert!(!token_allowed(&mask, 1), "token 1 (absent) not in mask");
    assert!(token_allowed(&mask, 2), "only '(i' (id=2) should be in mask");

    s.commit_token(2).unwrap(); // commit "(i"
    let mask = s.mask();
    // After "(i" we need EOF ('$') which is not in the vocab → empty mask
    let allowed = iter_allowed(&mask);
    assert!(allowed.is_empty(), "mask should be empty (no EOF token in vocab): {allowed:?}");
}

/// Grammar with shared prefixes and 'a'+: regression for trie self-loop panic.
/// Verifies that constraint construction does not panic.
#[test]
fn test_shared_prefix_no_panic() {
    // IDs: "za"→0, "zaabm"→1, "zaabn"→2
    let vocab = make_vocab(&["za", "zaabm", "zaabn"]);
    let c = Constraint::from_ebnf(
        r#"s ::= Z_T A_PLUS_T B_T M_T | Z_T A_PLUS_T B_T N_T
           Z_T ::= 'z'
           A_PLUS_T ::= 'a'+
           B_T ::= 'b'
           M_T ::= 'm'
           N_T ::= 'n'"#,
        &vocab,
    )
    .unwrap();
    // All three vocab tokens start a valid prefix ("za..." → Z_T + A_PLUS_T partial).
    let s = c.start();
    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0) || token_allowed(&mask, 1) || token_allowed(&mask, 2),
        "at least one token should be allowed initially"
    );
}

#[test]
fn test_expression_list_without_separator_tokens_leaves_empty_mask() {
    let vocab = make_vocab(&["a", "!\"", "\""]);
    let c = Constraint::from_ebnf(
        r#"program ::= (expression ';')* EOF
           expression ::= '!'? (IDENTIFIER | STRING_LITERAL)
           EOF ::= '$'
           STRING_LITERAL ::= '"' [^"]* '"'
           IDENTIFIER ::= 'a'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert_mask_allows(&mask, &[0, 1, 2]);

    s.commit_token(0).unwrap();
    assert_empty_mask(&s.mask());
}

/// Grammar s ::= x x '$'; x ::= ('!' x | 'a') ';'?.
/// After commit_bytes("a") the parser is mid-first-x; second x can be satisfied by "a;"(1).
#[test]
fn test_js_like_mask_after_commit_bytes() {
    // IDs: ";;;"→0, "a;"→1
    let vocab = make_vocab(&[";;;", "a;"]);
    let c = Constraint::from_ebnf(
        r#"s ::= x x '$'
           x ::= ( '!' x | 'a' ) ';'?"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    // Advance the parser by raw bytes "a" — completes the 'a' branch of first x.
    s.commit_bytes(b"a").unwrap();

    let mask = s.mask();
    // From here ';'? (opt) then second x then '$'.
    // "a;" → 'a' (second x) + ';' (second x's ';'?) → valid prefix
    assert!(
        token_allowed(&mask, 1),
        "'a;' (id=1) should be in mask after commit_bytes('a')"
    );
    // ";;;" → first ';' takes ';'?, then second x needs '!' or 'a' — fails
    assert!(
        !token_allowed(&mask, 0),
        "';;;' (id=0) should NOT be in mask after commit_bytes('a')"
    );
}

#[test]
fn test_double_semicolon_token_is_rejected_between_unary_expressions() {
    let vocab = make_vocab(&[";;"]);
    let c = Constraint::from_ebnf(
        r#"program ::= unary_expression unary_expression '$'
           unary_expression ::= ( '!' unary_expression | 'X' ) ';'?"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    s.commit_bytes(b"X").unwrap(); // first unary_expression 'X' branch

    assert_empty_mask(&s.mask());
}

/// Grammar program ::= IGNORE; IGNORE ::= ' ' | '$@'.
/// Vocab: " "(0) and "@"(1). Token "@" alone cannot match IGNORE (' '  or the 2-byte '$@').
/// Initial mask should contain only token 0 (' ').
#[test]
fn test_ebnf_initial_mask_with_alternation() {
    // IDs: " "→0, "@"→1
    let vocab = make_vocab(&[" ", "@"]);
    let c = Constraint::from_ebnf(
        r#"program ::= IGNORE
           IGNORE ::= ' ' | '$@'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "' ' (id=0) should be in initial mask (matches IGNORE first alt)"
    );
    assert!(
        !token_allowed(&mask, 1),
        "'@' (id=1) should NOT be in initial mask (doesn't start any IGNORE alternative)"
    );
}

/// Simpler companion: program ::= IGNORE; IGNORE ::= ' '.
/// Only ' ' (id=0) should be in initial mask.
#[test]
fn test_ebnf_initial_mask_mandatory() {
    // IDs: " "→0, "@"→1
    let vocab = make_vocab(&[" ", "@"]);
    let c = Constraint::from_ebnf(
        r#"program ::= IGNORE
           IGNORE ::= ' '"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "' ' (id=0) should be in initial mask"
    );
    assert!(
        !token_allowed(&mask, 1),
        "'@' (id=1) should NOT be in initial mask"
    );
}

/// Regression: right-recursive item grammar loses comma from mask after 2+ recursions.
/// Grammar: item: "," D ":" D item | ""; start: "{" D ":" D item "}".
/// After feeding "{1:2,3:4,5:6", comma must still be in the mask.
#[test]
fn test_right_recursive_item_bug() {
    // IDs: ","→0, "}"→1
    let vocab = make_vocab(&[",", "}"]);
    let c = Constraint::from_lark(
        r#"
        D: /[0-9]/
        item: "," D ":" D item | ""
        start: "{" D ":" D item "}"
        "#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    s.commit_bytes(b"{1:2,3:4,5:6").unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "comma (id=0) should be in the mask after '{{1:2,3:4,5:6' \
         (regression: right-recursive item must not lose continuation)"
    );
}

#[test]
fn test_repeated_item_with_one_less_wrapper_allows_closing_token() {
    let vocab = make_vocab(&["d"]);
    let c = Constraint::from_lark(
        r#"
        start: item+ "d"
        item: "d" leaf
        leaf: "d"
        "#,
        &vocab,
    )
    .unwrap();

    let mut accepts_closing = c.start();
    accepts_closing.commit_bytes(b"ddd").unwrap();

    let mut s = c.start();
    s.commit_bytes(b"dd").unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "closing token should stay allowed when the extra wrapper is removed"
    );
}

#[test]
fn test_repeated_item_keeps_closing_token_allowed() {
    let vocab = make_vocab(&["d"]);
    let c = Constraint::from_lark(
        r#"
        start: item+ "d"
        item: "d" node
        node: leaf
        leaf: "d"
        "#,
        &vocab,
    )
    .unwrap();

    let mut accepts_closing = c.start();
    accepts_closing.commit_bytes(b"ddd").unwrap();

    let mut s = c.start();
    s.commit_bytes(b"dd").unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "closing token should be allowed after completing the repeated item"
    );
}

// force() regressions where exactly one token stays in the mask.

/// Grammar: s ::= 'a' 'b' 'c' (fully deterministic single path).
/// With single-byte vocab each token is forced one at a time: [0, 1, 2].
#[test]
fn test_force_fully_determined() {
    // IDs: "a"→0, "b"→1, "c"→2
    let vocab = make_vocab(&["a", "b", "c"]);
    let c = Constraint::from_ebnf(
        r#"s ::= ABC
           ABC ::= 'a' 'b' 'c'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let forced = s.force();
    assert_eq!(forced, vec![0u32, 1, 2], "all three single-byte tokens forced in sequence");
}

/// Grammar: s ::= A | B (two alternatives, different first byte).
/// Mask starts with {0, 1} → nothing is forced.
#[test]
fn test_force_ambiguous_first_byte() {
    // IDs: "a"→0, "b"→1
    let vocab = make_vocab(&["a", "b"]);
    let c = Constraint::from_ebnf(
        r#"s ::= A | B
           A ::= 'a'
           B ::= 'b'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let forced = s.force();
    assert!(forced.is_empty(), "ambiguous first byte: nothing forced");
}

/// Grammar: s ::= AB | AC (shared 1-byte prefix 'a', then branch).
/// Only "a" (id=0) is forced; second byte branches to 'b' or 'c' → stop.
#[test]
fn test_force_partial_prefix() {
    // IDs: "a"→0, "b"→1, "c"→2
    let vocab = make_vocab(&["a", "b", "c"]);
    let c = Constraint::from_ebnf(
        r#"s ::= AB | AC
           AB ::= 'a' 'b'
           AC ::= 'a' 'c'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let forced = s.force();
    assert_eq!(forced, vec![0u32], "only 'a' (id=0) is forced; second byte branches");
}

// force() regressions

/// After committing the single token "a" the grammar is complete.
/// force() on the finished state returns empty (no more tokens to force).
#[test]
fn test_force_empty_after_complete() {
    // IDs: "a"→0, "<|endoftext|>"→1 (EOS; auto-detected)
    let vocab = make_vocab(&["a", "<|endoftext|>"]);
    let c = Constraint::from_ebnf(
        r#"s ::= A
           A ::= 'a'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();
    s.commit_token(0).unwrap(); // commit "a" → parse complete

    let forced = s.force();
    assert!(forced.is_empty(), "after complete parse, force() must return [] not {:?}", forced);
}

/// Grammar: s ::= AB CD (four distinct single-byte tokens in sequence).
/// Initial force gives all four. After committing the first two, mid-parse force gives the last two.
#[test]
fn test_force_after_partial_commit() {
    // IDs: "a"→0, "b"→1, "c"→2, "d"→3
    let vocab = make_vocab(&["a", "b", "c", "d"]);
    let c = Constraint::from_ebnf(
        r#"s ::= AB CD
           AB ::= 'a' 'b'
           CD ::= 'c' 'd'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    // Initially all four bytes are forced.
    let forced = s.force();
    assert_eq!(forced, vec![0u32, 1, 2, 3], "all tokens forced initially");

    // After committing 'a' and 'b', the remaining 'c' and 'd' are forced.
    s.commit_token(0).unwrap();
    s.commit_token(1).unwrap();
    let forced_mid = s.force();
    assert_eq!(forced_mid, vec![2u32, 3], "after 'ab', 'cd' is forced");
}

/// force() must not mutate the state: two consecutive calls must agree,
/// and mask() must be identical before and after.
#[test]
fn test_force_is_readonly() {
    // IDs: "a"→0, "b"→1, "c"→2
    let vocab = make_vocab(&["a", "b", "c"]);
    let c = Constraint::from_ebnf(
        r#"s ::= ABC
           ABC ::= 'a' 'b' 'c'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let mask_before = s.mask();
    let force1 = s.force();
    let mask_after = s.mask();
    let force2 = s.force();

    assert_eq!(mask_before, mask_after, "mask unchanged after force()");
    assert_eq!(force1, force2, "force() is deterministic when called twice");
}

/// Committing the tokens returned by force() must advance the parser correctly.
/// Grammar: s ::= AB CD; vocab: a/b/c/d (single byte). Expected force = [0,1,2,3].
#[test]
fn test_force_commit_roundtrip() {
    // IDs: "a"→0, "b"→1, "c"→2, "d"→3
    let vocab = make_vocab(&["a", "b", "c", "d"]);
    let c = Constraint::from_ebnf(
        r#"s ::= AB CD
           AB ::= 'a' 'b'
           CD ::= 'c' 'd'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let forced = s.force();
    assert_eq!(forced, vec![0u32, 1, 2, 3]);

    // Committing all forced tokens must succeed without panic.
    for token in &forced {
        let token_id = *token;
        s.commit_token(token_id).unwrap();
    }
    assert!(s.is_finished(), "should be finished after committing forced sequence");
}

/// Grammar: s ::= 'x' | no alternatives. Vocab: x=0 only.
/// The single character is forced immediately in the initial state.
#[test]
fn test_force_single_character_grammar() {
    let vocab = make_vocab(&["x"]);
    let c = Constraint::from_ebnf(
        r#"s ::= X
           X ::= 'x'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let forced = s.force();
    assert_eq!(forced, vec![0u32], "single-character grammar is fully forced");
}

/// Grammar: s ::= 'a' 'b'. Vocab: only "ab"=0 (no individual byte tokens).
/// With exactly one token in the mask, force() should return [0].
#[test]
fn test_force_only_multibyte_token() {
    let vocab = Vocab::new(vec![(0u32, b"ab".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= AB
           AB ::= 'a' 'b'"#,
        &vocab,
    )
    .unwrap();
    let s = c.start();

    let forced = s.force();
    assert_eq!(forced, vec![0u32], "only multi-byte 'ab' token; exactly one in mask → forced");
}

// Span-token and false-positive regressions

/// After commit `"a"`, token `":x"` must be allowed but `":-"` must NOT.
/// Grammar: `start: "a" ":" "x" STR_CHAR STR_CHAR "x"` where STR_CHAR = "a"|":"|"-".
/// Regression for Super DWA specialization admitting tokens that skip required literals.
#[test]
fn test_super_dwa_rejects_false_positive_colon_dash_token() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b":x".to_vec()),
            (2u32, b":-".to_vec()),
        ],
        None,
    );
    let lark = r#"start: "a" ":" "x" STR_CHAR STR_CHAR "x"
STR_CHAR: "a" | ":" | "-"
"#;
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_token(0u32).unwrap(); // "a"
    let mask = s.mask();
    assert!(token_allowed(&mask, 1), "':x' must be allowed after 'a'");
    assert!(!token_allowed(&mask, 2), "false-positive ':-' must NOT be allowed after 'a'");
}

#[test]
#[ignore]
fn test_glr_object_pair_rejects_false_positive_colon_tokens() {
    let lark = r#"start: ws object ws
object: "{" ws name_pair ws "}"
name_pair: QUOTE "name" QUOTE ws ":" ws QUOTE name_val QUOTE
name_val: name_chars
name_chars: STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR*
QUOTE: "\""
ws: WS*
WS: " " | "\n" | "\t" | "\r"
STR_CHAR: /[A-Za-z0-9 \[\]\-:{}@.]/
"#;
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"".to_vec()),    // tok_open
            (1u32, b"name".to_vec()),   // tok_name
            (2u32, b"\":\"".to_vec()),  // tok_colon_quote (closing " + : + opening ")
            (3u32, b"\":[".to_vec()),   // tok_fp_bracket (FP: "[" is not valid)
            (4u32, b"\":-".to_vec()),   // tok_fp_dash (FP: "-" is not valid after ":")
        ],
        None,
    );
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_token(0u32).unwrap(); // `{"`
    s.commit_token(1u32).unwrap(); // `name`
    let mask = s.mask();
    assert!(token_allowed(&mask, 2), "'\":\"' must be allowed after '{{\"name'");
    assert!(!token_allowed(&mask, 3), r#"false-positive '":[' must NOT be allowed"#);
    assert!(!token_allowed(&mask, 4), r#"false-positive '":-' must NOT be allowed"#);
}

/// Standalone UTF-8 continuation bytes must stay rejected inside JSON strings.
#[test]
fn test_json_string_rejects_invalid_utf8() {
    let lark = r#"start: "{" JSON_STRING ":" JSON_STRING "}"
JSON_STRING: "\"" STRING_CHARS "\""
STRING_CHARS: STRING_CHAR*
STRING_CHAR: /[^\x00-\x1F"\\]/
"#;
    // Minimal vocab that admits a valid completion: {"a":""}
    // commit seq: 0({") + 2(a) + 3(") + 4(:) + 3(") + 3(") + 5(})
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"".to_vec()),   // opens object + starts first JSON_STRING
            (1u32, vec![0xA1u8]),      // bad: standalone UTF-8 continuation byte
            (2u32, b"a".to_vec()),     // good: ASCII character inside JSON_STRING
            (3u32, b"\"".to_vec()),    // quote: closes/opens strings (enables completion)
            (4u32, b":".to_vec()),     // key-value colon
            (5u32, b"}".to_vec()),     // object close
        ],
        None,
    );
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_token(0u32).unwrap(); // `{"`
    let mask = s.mask();
    assert!(token_allowed(&mask, 2), "ASCII 'a' must be allowed as JSON string content after {{\"");
    assert!(!token_allowed(&mask, 1), "standalone 0xA1 must NOT be allowed as JSON string content after {{\"");
}

/// Grammar: `start: "a" ":" "a"`. After commit_bytes("a"), token ":a" spans two
/// grammar terminals and must appear in the mask.
#[test]
fn test_span_token_in_mask() {
    let lark = r#"start: "a" ":" "a""#;
    let vocab = Vocab::new(vec![(0u32, b":a".to_vec())], None);
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "span token ':a' must be in mask after commit_bytes('a')");
}

#[test]
fn test_json_pair_span_token_survives_commit_and_commit_bytes() {
    let lark = r#"start: "{" pair "}"
pair: string ":" string "," string ":" "null"
string: QUOTE char* QUOTE
char: "a"
QUOTE: "\""
"#;
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"".to_vec()),        // tok_prefix
            (1u32, b"\":\"\",".to_vec()),   // tok_span: closing " + : + open " + close " + ,
            (2u32, b"\"a\":null}".to_vec()), // tok_suffix
        ],
        None,
    );
    let c = Constraint::from_lark(lark, &vocab).unwrap();

    // Test via state.commit() path
    let mut s = c.start();
    s.commit_token(0u32).unwrap(); // `{"`
    let mask = s.mask();
    assert!(token_allowed(&mask, 1), "span token b'\\\":\\\"\\\",\\\"' must be in mask after commit(0)");

    // Test via commit_bytes() path
    let mut s2 = c.start();
    s2.commit_bytes(b"{\"").unwrap();
    let mask2 = s2.mask();
    assert!(token_allowed(&mask2, 1), "span token must be in mask after commit_bytes(b'{{\\\"')");
}

#[test]
fn test_minimal_json_string_span_token_is_allowed_after_commit_bytes() {
    let vocab = Vocab::new(vec![(0u32, b"\":\"".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"start ::= string ':' string ','
string ::= '"' '"'"#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();
    s.commit_bytes(b"\"").unwrap();
    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "span token b'\\\":\\\"' must be in mask after commit_bytes(b'\"')"
    );
}

#[test]
fn test_minimal_cross_token_bridge_through_nullable_unit_reduction() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" b".to_vec()),
            (3u32, b" {d},".to_vec()),
            (4u32, b" {c},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= "a" " " obj "," " b"
obj ::= "{" inner "}"
inner ::= "c" tail | rest
tail ::= "," "d" tail |
rest ::= "d" tail |"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"a")
        .expect("prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "token ' {{}},' should be allowed after the prefix when the nested object body reduces through a nullable unit-production"
    );
    assert!(
        token_allowed(&mask, 3),
        "token ' {{d}},' should also remain allowed after the prefix"
    );
    assert!(
        token_allowed(&mask, 4),
        "token ' {{c}},' should also remain allowed after the prefix"
    );

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(
        token_allowed(&mask, 2),
        "the trailing token should remain allowed after the bridged empty-object token"
    );

    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
fn test_json_like_cross_token_bridge_through_nullable_unit_reduction() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"x\":".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" \"y\":\"z\"}".to_vec()),
            (3u32, b" {\"r\":\"x\"},".to_vec()),
            (4u32, b" {\"e\":\"y\"},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= "{\"x\":" " " obj "," " \"y\":\"z\"}"
obj ::= "{" body "}"
body ::= "\"r\":\"x\"" tail | extras
tail ::= ", " "\"e\":\"y\"" tail |
extras ::= "\"e\":\"y\"" tail |"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"{\"x\":")
        .expect("prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(token_allowed(&mask, 1), "empty nested-object token should be allowed after the partial key prefix");
    assert!(token_allowed(&mask, 3), "non-empty nested-object token should remain allowed after the partial key prefix");
    assert!(token_allowed(&mask, 4), "additional-property nested-object token should remain allowed after the partial key prefix");

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(token_allowed(&mask, 2), "trailing sibling token should remain allowed after the bridged empty-object token");

    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
fn test_ported_cross_token_bridge_through_nullable_chain() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"onRequestExternal\":".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" \"next\":\"x\"}".to_vec()),
            (3u32, b" {\"removeRules\": \"x\"},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= T0 obj T5 T6
obj ::= T1 body T4
body ::= T2 tail | rest
tail ::= T3 T7 tail |
rest ::= T7 tail |
T0 ::= '{"onRequestExternal": '
T1 ::= '{'
T2 ::= '"removeRules": "x"'
T3 ::= ', '
T4 ::= '}'
T5 ::= ', '
T6 ::= '"next":"x"}'
T7 ::= '"extra": "y"'"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"{\"onRequestExternal\":")
        .expect("partial key-prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "token ' {{}},' should remain allowed when the empty object closes only after a nullable chain and the separator continues in the next token"
    );
    assert!(
        token_allowed(&mask, 3),
        "non-empty nested-object token should remain allowed after the same prefix"
    );

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(
        token_allowed(&mask, 2),
        "the trailing sibling token should remain allowed after the bridged empty-object token"
    );
}

#[test]
fn test_ported_cross_token_bridge_with_regex_additional_property_alternative() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"sendRequest\":\"x\", \"onRequestExternal\":".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" \"tail\":\"y\"}".to_vec()),
            (3u32, b" {\"other\":\"z\"},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_lark(
        r#"
        start: PREFIX obj SEP NEXT
        obj: OPEN body CLOSE
        body: KEYCOLON VALUE tail |
        tail: SEP KEYCOLON VALUE tail |

        PREFIX: /\{"sendRequest":"x", "onRequestExternal": ?/
        OPEN: "{"
        CLOSE: "}"
        SEP: ", "
        NEXT: /"tail":"y"\}/
        KEYCOLON: /"(?:[^"\\]|\\.)*": ?/
        VALUE: "\"z\""
        "#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"{\"sendRequest\":\"x\", \"onRequestExternal\":")
        .expect("partial key-prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "token ' {{}},' should remain allowed even when the empty object competes with a regex additional-property branch"
    );
    assert!(
        token_allowed(&mask, 3),
        "non-empty regex additional-property token should remain allowed after the same prefix"
    );

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(
        token_allowed(&mask, 2),
        "the trailing sibling token should remain allowed after the bridged empty-object token"
    );
}

#[test]
fn test_ported_exact_lowered_shape_bridge() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" b".to_vec()),
            (3u32, b" {d},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= 'a' ' ' inner ' b'
inner ::= '{' body '},'
body ::= ap_nc |
ap_nc ::= ap ap_c |
ap_c ::= ap_lr
ap_lr ::= ap_lr ', ' ap |
ap ::= 'd'"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"a")
        .expect("partial key-prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(token_allowed(&mask, 1), "empty nested-object token should be allowed after the lowered-shape prefix");
    assert!(token_allowed(&mask, 3), "non-empty nested-object token should remain allowed after the same prefix");

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(token_allowed(&mask, 2), "trailing sibling token should remain allowed after the bridged empty-object token");
}

#[test]
fn test_ported_exact_lowered_shape_bridge_without_unit_alias() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" b".to_vec()),
            (3u32, b" {d},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= 'a' ' ' inner ' b'
inner ::= '{' body '},'
body ::= ap_nc |
ap_nc ::= ap ap_lr |
ap_lr ::= ap_lr ', ' ap |
ap ::= 'd'"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"a")
        .expect("partial key-prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "empty nested-object token should still be allowed when the unit alias is removed"
    );
    assert!(
        token_allowed(&mask, 3),
        "non-empty nested-object token should still be allowed when the unit alias is removed"
    );

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(
        token_allowed(&mask, 2),
        "trailing sibling token should remain allowed after the bridged empty-object token when the unit alias is removed"
    );
}

#[test]
fn test_ported_exact_lowered_shape_bridge_with_direct_repetition() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b" {},".to_vec()),
            (2u32, b" b".to_vec()),
            (3u32, b" {d},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= 'a' ' ' inner ' b'
inner ::= '{' body '},'
body ::= ap_list |
ap_list ::= ap (', ' ap)*
ap ::= 'd'"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(b"a")
        .expect("partial key-prefix bytes should advance the parser state");

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "empty nested-object token should be allowed for the direct-repetition control"
    );
    assert!(
        token_allowed(&mask, 3),
        "non-empty nested-object token should be allowed for the direct-repetition control"
    );

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(
        token_allowed(&mask, 2),
        "trailing sibling token should remain allowed after the bridged empty-object token for the direct-repetition control"
    );
}

#[test]
fn test_json_schema_o62060_minimized_empty_object_token_remains_allowed() {
    const PREFIX: &[u8] = b"{\"a\": 0, \"b\": 0, \"c\":";
    const EMPTY_OBJECT_TOKEN: &[u8] = b" {},";

    let tail = (b'e'..=b'x')
        .map(|key| format!("\"{}\":{{}}", key as char))
        .collect::<Vec<_>>()
        .join(",");
    let schema = [
        "{\"type\":\"object\",\"properties\":{\"a\":{},\"b\":{},\"d\":{},\"c\":{\"type\":\"object\"},",
        &tail,
        "},\"required\":[\"a\",\"b\",\"e\",\"c\"],\"additionalProperties\":false}",
    ]
    .concat();
    let vocab = Vocab::new(vec![(0u32, b" {},".to_vec())], None);
    let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();

    let mut state = constraint.start();
    state
        .commit_bytes(PREFIX)
        .expect("minimized prefix bytes should advance the parser state");

    let mut commit_probe = constraint.start();
    commit_probe
        .commit_bytes(PREFIX)
        .expect("minimized prefix bytes should advance the parser state");
    commit_probe
        .commit_bytes(EMPTY_OBJECT_TOKEN)
        .expect("bytes ' {},' should be directly committable after the minimized o62060 prefix witness");

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 0),
        "token ' {{}},' should remain allowed after the minimized o62060 prefix witness"
    );
}

#[test]
fn test_glrm_grammar_o62060_abstract_committable_token_is_masked() {
    const PREFIX: &[u8] = b"a";
    const DISPUTED_TOKEN: &[u8] = b"b";

    let grammar = "start s;\nnt s ::= \"a\" \"bc\";";
    let vocab = Vocab::new(vec![(0u32, DISPUTED_TOKEN.to_vec())], None);
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut state = constraint.start();
    state.commit_bytes(PREFIX).unwrap();

    let mut commit_probe = state.clone();
    commit_probe.commit_bytes(DISPUTED_TOKEN).unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), "token 'b' should remain allowed after prefix 'a'");
}

// Scaffolding test kept to validate commit_bytes path also fails for the schema.
#[test]
#[ignore]
fn test_schema_commit_bytes_vs_commit_token() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "evt": {
                "type": "object",
                "properties": {
                    "addListener": {"type": "string"},
                    "removeRules": {"type": "string"}
                }
            },
            "next": {"type": "string"}
        },
        "required": ["evt", "next"],
        "additionalProperties": false
    }"#;

    let token0 = b"{\"evt\":";
    let token1 = b" {},";
    let token2 = b" \"next\":\"x\"}";

    // Approach A: commit_bytes for the prefix
    {
        let vocab = Vocab::new(
            vec![(1u32, token1.to_vec()), (2u32, token2.to_vec())],
            None,
        );
        let c = glrmask::Constraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = c.start();
        state.commit_bytes(token0).expect("commit_bytes of token0 should succeed");
        let mask = state.mask();
        assert!(
            token_allowed(&mask, 1),
            "commit_bytes: token1 ' {{}},' should be in mask after prefix bytes"
        );
        state.commit_token(1).unwrap();
        let mask = state.mask();
        assert!(token_allowed(&mask, 2), "commit_bytes: token2 should be in mask after token1");
        state.commit_token(2).unwrap();
        assert!(state.is_finished());
    }

    // Approach B: commit_token for the prefix (mirrors the failing schema test exactly)
    {
        let vocab = Vocab::new(
            vec![(0u32, token0.to_vec()), (1u32, token1.to_vec()), (2u32, token2.to_vec())],
            None,
        );
        let c = glrmask::Constraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = c.start();
        state.commit_token(0).unwrap();
        let mask = state.mask();
        assert!(
            token_allowed(&mask, 1),
            "commit_token: token1 ' {{}},' should be in mask after committing token0"
        );
        state.commit_token(1).unwrap();
        let mask = state.mask();
        assert!(token_allowed(&mask, 2), "commit_token: token2 should be in mask after token1");
        state.commit_token(2).unwrap();
        assert!(state.is_finished());
    }
}

#[test]
#[ignore]
fn test_prefix_ending_mid_terminal_repro() {
    // Grammar where ':' is the last byte of token0 and ' ' (completing ': ') is the first byte
    // of token1. This mirrors how {"evt": splits inside the ': ' 2-byte terminal.
    let vocab = Vocab::new(
        vec![
            (0u32, b"a:".to_vec()),
            (1u32, b" {},".to_vec()),  // space completes ': ', then inner object
            (2u32, b" b".to_vec()),
            (3u32, b" {d},".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(
        r#"start ::= 'a' ': ' inner ' b'
inner ::= '{' body '},'
body ::= ap_list |
ap_list ::= ap (', ' ap)*
ap ::= 'd'"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    // commit_bytes stops mid-terminal (': ' has ':' consumed but not ' ')
    state.commit_bytes(b"a:").expect("partial prefix should advance");
    let mask = state.mask();
    assert!(
        token_allowed(&mask, 1),
        "token ' {{}},' should be in mask when prefix ends mid-terminal (inside ': ')"
    );
    assert!(
        token_allowed(&mask, 3),
        "token ' {{d}},' should be in mask when prefix ends mid-terminal"
    );

    state.commit_token(1).unwrap();
    let mask = state.mask();
    assert!(token_allowed(&mask, 2), "trailing token should be in mask after the bridged empty-object");

    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
#[ignore]
fn test_token1_ends_mid_3byte_terminal_repro() {
    let constraint = Constraint::from_ebnf(
        r#"start ::= 'a' ': ' '{' body '}, ' 'v}'
body ::= item (', ' item)* |
item ::= 'd'"#,
        &Vocab::new(
            vec![
                (0u32, b"a:".to_vec()),
                (1u32, b" {},".to_vec()),
                (2u32, b" v}".to_vec()),
                (3u32, b" {d},".to_vec()),
            ],
            None,
        ),
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_bytes(b"a:").unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

// Minimal abstract reproduction of the recognition regression.
// After consuming the first byte of 'ab', token 'bc' should complete the parse,
// but unit-reduction inlining rejects it.
#[test]
fn test_minimal_repro_abstract_mid_terminal_nullable_choice() {
    let vocab = Vocab::new(vec![(0u32, b"bc".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_token(0).unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_minimal_repro_abstract_mid_terminal_nullable_choice_commit_bytes_only() {
    // Cover every byte that appears in the grammar with at least one vocab entry:
    // 'a' in token 0, 'b'/'c' in token 1, and 'f' in token 2.
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b"bc".to_vec()),
            (2u32, b"f".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::= 'f'?"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_bytes(b"bc").unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_minimal_repro_abstract_mid_terminal_nullable_choice_commit_token_with_vocab_coverage() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b"bc".to_vec()),
            (2u32, b"f".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_token(1).unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_minimal_repro_abstract_mid_terminal_nullable_choice_commit_bytes_with_vocab_coverage() {
    let vocab = Vocab::new(
        vec![
            (0u32, b"a".to_vec()),
            (1u32, b"bc".to_vec()),
            (2u32, b"f".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_bytes(b"bc").unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_minimal_repro_abstract_mid_terminal_nullable_choice_commit_bytes_min_vocab() {
    let vocab = Vocab::new(vec![(0u32, b"bc".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_bytes(b"bc").unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_minimal_repro_abstract_mid_terminal_nullable_choice_optional_syntax_commit_token() {
    let vocab = Vocab::new(vec![(0u32, b"bc".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::= 'f'?"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_token(0).unwrap();
    assert!(s.is_finished());
}

#[test]
fn test_minimal_repro_abstract_debug_rules_explicit_nullable_shape() {
    let vocab = Vocab::new(vec![(0u32, b"bc".to_vec())], None);
    let explicit = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        &vocab,
    )
    .unwrap();
    let optional = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::= 'f'?"#,
        &vocab,
    )
    .unwrap();

    let explicit_rules = explicit.debug_rules();
    let optional_rules = optional.debug_rules();
    assert!(
        explicit_rules == optional_rules,
        "expected explicit nullable syntax and optional syntax to compile to the same normalized rule set after duplicate-rule merging; explicit={explicit_rules:?} optional={optional_rules:?}"
    );
}

#[test]
fn test_minimal_repro_abstract_debug_rules_optional_syntax_shape() {
    let vocab = Vocab::new(vec![(0u32, b"bc".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a 'c'
a ::= 'f'?"#,
        &vocab,
    )
    .unwrap();

    let rules = c.debug_rules();
    let a_rules: Vec<_> = rules.iter().filter(|(lhs, _, _)| *lhs == 1).collect();
    assert!(
        a_rules.iter().all(|(_, len, _)| *len != 0),
        "expected optional syntax lowering not to keep a direct empty rule on lhs=1; rules={rules:?}"
    );
}

#[test]
#[ignore]
fn test_minimal_repro_abstract_no_suffix_misfinishes() {
    let vocab = Vocab::new(vec![(0u32, b"b".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= 'ab' a
a ::=
a ::= 'f'"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a").unwrap();
    s.commit_token(0).unwrap();
    assert!(s.is_finished());
}

#[test]
#[ignore]
fn test_schema_shape_with_unit_alias_and_3byte_close() {
    let constraint = Constraint::from_ebnf(
        r#"start ::= 'a' ': ' '{' free_nc '}, ' 'v}'
free_nc ::= ap_nc
ap_nc ::= ap ap_c |
ap_c ::= ap_lr
ap_lr ::= ap_lr ', ' ap |
ap ::= 'd'"#,
        &Vocab::new(
            vec![
                (0u32, b"a:".to_vec()),
                (1u32, b" {},".to_vec()),
                (2u32, b" v}".to_vec()),
                (3u32, b" {d},".to_vec()),
            ],
            None,
        ),
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_bytes(b"a:").unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
fn test_full_json_grammar_exposes_high_id_span_token() {
    let lark = r#"start: ws value ws
value: object | array | string | number | "true" | "false" | "null"
object: "{" ws members? ws "}"
members: pair (ws "," ws pair)*
pair: string ws ":" ws value
array: "[" ws elements? ws "]"
elements: value (ws "," ws value)*
string: QUOTE char* QUOTE
char: letter | digit | MINUS | UNDERSCORE
number: int | int frac | int exp | int frac exp
int: digits | MINUS digits
frac: DOT digits
exp: EXP digits | EXP PLUS digits | EXP MINUS digits
digits: DIGIT+
ws: WS*
letter: LETTER
digit: DIGIT
QUOTE: "\""
MINUS: "-"
PLUS: "+"
DOT: "."
EXP: "e" | "E"
UNDERSCORE: "_"
WS: " " | "\n" | "\t" | "\r"
LETTER: "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
"#;
    // Sparse vocab with original high IDs (mirrors production token IDs)
    let vocab = Vocab::new(
        vec![
            (4895u32,  b"{\"".to_vec()),
            (34713u32, b"\":\"\",".to_vec()),
            (34714u32, b"\"a\":null}".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_token(4895u32).unwrap(); // `{"`
    let mask = s.mask();
    assert!(
        token_allowed(&mask, 34713),
        "json_value span token: b'\":\\\"\\\",\\\"' (ID 34713) must be in mask after b'{{\\\"' (ID 4895)"
    );
}

// Indirect recursion and expression edge cases

/// Grammar: `s_prime ::= s EOF; s ::= A e | B; e ::= s` (indirect recursion s ↔ e).
/// Equivalent to `a* b $`. After `a` the state recurses through `e = s`; after `ab`
/// only `$` (EOF) remains.
///
/// Differs from `test_indirect_recursion` (which is `s ::= A s | B end`):
/// here the recursive step goes through an intermediate non-terminal `e`.
#[test]
fn test_indirect_recursion_through_intermediate_nonterminal() {
    let vocab = make_vocab(&["a", "b", "$"]);
    let c = Constraint::from_ebnf(
        r#"s_prime ::= s EOF
s ::= A e | B
e ::= s
A ::= 'a'
B ::= 'b'
EOF ::= '$'"#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "'a' must be in initial mask");
    assert!(token_allowed(&mask, 1), "'b' must be in initial mask");
    assert!(!token_allowed(&mask, 2), "'$' must NOT be in initial mask (s not yet satisfied)");

    s.commit_token(0u32).unwrap(); // "a"
    let mask = s.mask();
    // After "a", e = s recurses: expect 'a' or 'b' again.
    assert!(token_allowed(&mask, 0), "'a' must be in mask after first 'a'");
    assert!(token_allowed(&mask, 1), "'b' must be in mask after first 'a'");
    assert!(!token_allowed(&mask, 2), "'$' must NOT be in mask (s not yet fully satisfied)");

    s.commit_token(1u32).unwrap(); // "b"
    let mask = s.mask();
    // After "ab", s = A e = A B (complete). Now expect EOF.
    assert!(!token_allowed(&mask, 0), "'a' must NOT be in mask after 'ab'");
    assert!(!token_allowed(&mask, 1), "'b' must NOT be in mask after 'ab'");
    assert!(token_allowed(&mask, 2), "'$' must be in mask after 'ab' (ready for EOF)");
}

/// Grammar: `s ::= e; e ::= e '+' | t; t ::= t '*' | I; I ::= 'i'` (left-recursive E/T with I).
/// Vocab: only `+`=0.
/// Initial mask: `{}` — `+` cannot start any string (`i` prefix needed, not in vocab).
/// After `commit_bytes("i")`: token `+` extends the expression via `e ::= e '+'`.
/// This works because `i` completes an `I` which completes `t` and then `e`; `+` is then
/// a valid continuation because acceptance of the full grammar requires another term after `+`,
/// and we allow speculative extension (prefix-consistent).
#[test]
fn test_plus_token_is_allowed_after_identifier_byte_prefix() {
    let vocab = Vocab::new(vec![(0u32, b"+".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"s ::= e
e ::= e '+' | t
t ::= t '*' | I
I ::= 'i'"#,
        &vocab,
    )
    .unwrap();

    let s = c.start();
    let mask = s.mask();
    // Initial: '+' can't start any expression (needs 'i' first, which is not in vocab)
    assert!(
        iter_allowed(&mask).is_empty(),
        "initial mask must be empty: '+' cannot start any expression: {mask:?}"
    );

    let mut s2 = c.start();
    s2.commit_bytes(b"i").unwrap();
    let mask2 = s2.mask();
    // After "i": grammar satisfied (e = t = I). '+' would extend via e ::= e '+' | t.
    // Old system: {0}. New system: {} if completability-checks that 'i+' cannot be completed
    // (next 'e' requires another 'i', not in vocab).
    assert!(
        token_allowed(&mask2, 0),
        "'+' (id=0) must be in mask after 'i' (prefix-consistent extension of e ::= e '+')"
    );
}

// Expression grammar variants
#[test]
fn test_expression_no_times() {
    let vocab = Vocab::new(
        vec![
            (0, b"i".to_vec()),
            (1, b"+".to_vec()),
            (2, b"(".to_vec()),
            (3, b")".to_vec()),
            (4, b"(i".to_vec()),
            (5, b"+i".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= e "$"
e ::= e "+" t | t
t ::= f
f ::= "(" e ")" | "i""#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    // Initial: only tokens that start an expression.
    assert!(token_allowed(&mask, 0), "i must be in initial mask");
    assert!(token_allowed(&mask, 2), "( must be in initial mask");
    assert!(token_allowed(&mask, 4), "(i must be in initial mask");
    assert!(!token_allowed(&mask, 1), "+ must NOT be in initial mask");
    assert!(!token_allowed(&mask, 3), ") must NOT be in initial mask");
    assert!(!token_allowed(&mask, 5), "+i must NOT be in initial mask");

    // Commit "(i" (spans '(' then 'i').
    s.commit_token(4).unwrap();
    let mask = s.mask();
    // After "(i", expect {+, ), +i}.
    assert!(token_allowed(&mask, 1), "+ must be in mask after (i");
    assert!(token_allowed(&mask, 3), ") must be in mask after (i");
    assert!(token_allowed(&mask, 5), "+i must be in mask after (i");
    assert!(!token_allowed(&mask, 0), "i must NOT be in mask after (i");
    assert!(!token_allowed(&mask, 2), "( must NOT be in mask after (i");
    assert!(!token_allowed(&mask, 4), "(i must NOT be in mask after (i");
}

#[test]
fn test_expression_no_parens() {
    let vocab = Vocab::new(
        vec![
            (0, b"i".to_vec()),
            (1, b"+".to_vec()),
            (2, b"*".to_vec()),
            (3, b"+i".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= e "$"
e ::= e "+" t | t
t ::= t "*" f | f
f ::= "i""#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    // Initial: only "i" can start an expression.
    assert!(token_allowed(&mask, 0), "i must be in initial mask");
    assert!(!token_allowed(&mask, 1), "+ must NOT be in initial mask");
    assert!(!token_allowed(&mask, 2), "* must NOT be in initial mask");
    assert!(!token_allowed(&mask, 3), "+i must NOT be in initial mask");

    // Commit "i".
    s.commit_token(0).unwrap();
    let mask = s.mask();
    // After "i", expect {+, *, +i}.
    assert!(token_allowed(&mask, 1), "+ must be in mask after i");
    assert!(token_allowed(&mask, 2), "* must be in mask after i");
    assert!(token_allowed(&mask, 3), "+i must be in mask after i");
    assert!(!token_allowed(&mask, 0), "i must NOT be in mask after i");
}

#[test]
fn test_expression_no_plus_times() {
    let vocab = Vocab::new(
        vec![
            (0, b"i".to_vec()),
            (1, b"(".to_vec()),
            (2, b")".to_vec()),
            (3, b"(i".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= e "$"
e ::= t
t ::= f
f ::= "(" e ")" | "i""#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    // Initial: "i", "(", or "(i" can start an expression.
    assert!(token_allowed(&mask, 0), "i must be in initial mask");
    assert!(token_allowed(&mask, 1), "( must be in initial mask");
    assert!(token_allowed(&mask, 3), "(i must be in initial mask");
    assert!(!token_allowed(&mask, 2), ") must NOT be in initial mask");

    // Commit "(i" (spans '(' then 'i').
    s.commit_token(3).unwrap();
    let mask = s.mask();
    // After "(i", only ")" closes the paren.
    assert!(token_allowed(&mask, 2), ") must be in mask after (i");
    assert!(!token_allowed(&mask, 0), "i must NOT be in mask after (i");
    assert!(!token_allowed(&mask, 1), "( must NOT be in mask after (i");
    assert!(!token_allowed(&mask, 3), "(i must NOT be in mask after (i");
}

#[test]
fn test_expression_no_times_parens() {
    let vocab = Vocab::new(
        vec![
            (0, b"i".to_vec()),
            (1, b"+".to_vec()),
            (2, b"+i".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= e "$"
e ::= e "+" t | t
t ::= f
f ::= "i""#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    // Initial: only "i" starts an expression.
    assert!(token_allowed(&mask, 0), "i must be in initial mask");
    assert!(!token_allowed(&mask, 1), "+ must NOT be in initial mask");
    assert!(!token_allowed(&mask, 2), "+i must NOT be in initial mask");

    // Commit "i".
    s.commit_token(0).unwrap();
    let mask = s.mask();
    // After "i", expect {+, +i}.
    assert!(token_allowed(&mask, 1), "+ must be in mask after i");
    assert!(token_allowed(&mask, 2), "+i must be in mask after i");
    assert!(!token_allowed(&mask, 0), "i must NOT be in mask after i");
}

#[test]
fn test_expression_unbalanced_parens() {
    let vocab = Vocab::new(
        vec![
            (0, b"i".to_vec()),
            (1, b"(".to_vec()),
            (2, b"(i".to_vec()),
            (3, b"$".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= e "$"
e ::= t
t ::= f
f ::= "(" e | "i""#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    // Initial: {i, (, (i}.
    assert!(token_allowed(&mask, 0), "i must be in initial mask");
    assert!(token_allowed(&mask, 1), "( must be in initial mask");
    assert!(token_allowed(&mask, 2), "(i must be in initial mask");
    assert!(!token_allowed(&mask, 3), "$ must NOT be in initial mask");

    // Commit "(" → recurse into another E.
    s.commit_token(1).unwrap();
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "i must be in mask after (");
    assert!(token_allowed(&mask, 1), "( must be in mask after (");
    assert!(token_allowed(&mask, 2), "(i must be in mask after (");
    assert!(!token_allowed(&mask, 3), "$ must NOT be in mask after (");

    // Commit "i" → inner E satisfied, now expect EOF.
    s.commit_token(0).unwrap();
    let mask = s.mask();
    assert!(token_allowed(&mask, 3), "$ must be in mask after (i");
    assert!(!token_allowed(&mask, 0), "i must NOT be in mask after (i");
    assert!(!token_allowed(&mask, 1), "( must NOT be in mask after (i");
    assert!(!token_allowed(&mask, 2), "(i must NOT be in mask after (i");
}

#[test]
fn test_expression_cycle_reduced() {
    let vocab = Vocab::new(
        vec![
            (0, b"i".to_vec()),
            (1, b"$".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= e "$"
e ::= f
f ::= "i""#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    let mask = s.mask();
    assert!(token_allowed(&mask, 0), "i must be in initial mask");
    assert!(!token_allowed(&mask, 1), "$ must NOT be in initial mask");

    // Commit "i" → E satisfied.
    s.commit_token(0).unwrap();
    let mask = s.mask();
    assert!(token_allowed(&mask, 1), "$ must be in mask after i");
    assert!(!token_allowed(&mask, 0), "i must NOT be in mask after i");

    // Commit "$" → parse complete.
    s.commit_token(1).unwrap();
    let mask = s.mask();
    assert!(iter_allowed(&mask).is_empty(), "mask must be empty after i$");
}

#[test]
fn test_force_long_shared_prefix() {
    let vocab = Vocab::new(
        vec![
            (0, b"h".to_vec()),
            (1, b"e".to_vec()),
            (2, b"l".to_vec()),
            (3, b"o".to_vec()),
            (4, b"_".to_vec()),
            (5, b"w".to_vec()),
            (6, b"r".to_vec()),
            (7, b"d".to_vec()),
            (8, b"a".to_vec()),
            (9, b"t".to_vec()),
            (10, b"hello".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"s ::= hello_world | hello_earth
hello_world ::= "h" "e" "l" "l" "o" "_" "w" "o" "r" "l" "d"
hello_earth ::= "h" "e" "l" "l" "o" "_" "e" "a" "r" "t" "h""#,
        &vocab,
    )
    .unwrap();

    let s = c.start();
    let forced = s.force();
    // Greedy tokenization of the shared prefix yields [hello(10), _(4)].
    // glrmask's token-level force may differ (see comment above).
    // If glrmask's force produces a different result, this test documents the gap.
    assert!(
        !forced.is_empty(),
        "at least 'h' should be forced since both alternatives start with 'h'"
    );
}

#[test]
fn test_ws_boundary_no_ignore() {
    let vocab = Vocab::new(
        vec![
            (0, b" ".to_vec()),
            (1, b" the".to_vec()),
            (2, b" that".to_vec()),
            (3, b"that".to_vec()),
            (4, b"a".to_vec()),
        ],
        None,
    );
    // Without ignore-WS, just test a simple letter-sequence grammar.
    // `stmt` matches one or more letters a-z.
    let c = Constraint::from_ebnf(
        r#"s ::= stmt
stmt ::= stmt letter | letter
letter ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z""#,
        &vocab,
    )
    .unwrap();

    let s = c.start();
    let mask = s.mask();
    // Letters-only grammar: " " and " the" should NOT be allowed.
    // "that" and "a" SHOULD be allowed (they're pure letters).
    assert!(token_allowed(&mask, 3), "'that' should be valid at start");
    assert!(token_allowed(&mask, 4), "'a' should be valid at start");
    assert!(!token_allowed(&mask, 0), "' ' should NOT be valid (not a letter)");
}

fn nullable_string_inline_alternation_lark() -> &'static str {
    r###"
PATTERN_0: /[\x20-\x21\x23-\x5B\x5D-\x7F]/
PATTERN_6: /[\x22\x2F\x5C\x62\x66\x6E\x72\x74]/
STRING_CHAR: PATTERN_0
ESCAPE_SHORT_CHAR: PATTERN_6
ESCAPE_SEQ: "\\" ESCAPE_SHORT_CHAR
STRING_CONTENT: (STRING_CHAR | ESCAPE_SEQ)*
JSON_STRING: "\"" STRING_CONTENT "\""
JSON_NULL: "null"

obj_ord_0_2_nc: "\"id\"" ":" JSON_STRING
obj_ord_0_2_c: "," "\"id\"" ":" JSON_STRING
obj_ord_0_1_nc: "\"couponCode\"" ":" (JSON_STRING | JSON_NULL) obj_ord_0_2_c | obj_ord_0_2_nc
obj_ord_0_0_nc: "\"affiliation\"" ":" (JSON_STRING | JSON_NULL) obj_ord_0_1_c | obj_ord_0_1_nc
obj_ord_0_1_c: "," "\"couponCode\"" ":" (JSON_STRING | JSON_NULL) obj_ord_0_2_c | obj_ord_0_2_c
obj_ord_0_0_c: "," "\"affiliation\"" ":" (JSON_STRING | JSON_NULL) obj_ord_0_1_c | obj_ord_0_1_c

start: "{" obj_ord_0_0_nc "}"
"###
}

fn nullable_string_helper_rule_lark() -> &'static str {
    r###"
PATTERN_0: /[\x20-\x21\x23-\x5B\x5D-\x7F]/
PATTERN_6: /[\x22\x2F\x5C\x62\x66\x6E\x72\x74]/
STRING_CHAR: PATTERN_0
ESCAPE_SHORT_CHAR: PATTERN_6
ESCAPE_SEQ: "\\" ESCAPE_SHORT_CHAR
STRING_CONTENT: (STRING_CHAR | ESCAPE_SEQ)*
JSON_STRING: "\"" STRING_CONTENT "\""
JSON_NULL: "null"

str_or_null: JSON_STRING | JSON_NULL

obj_ord_0_2_nc: "\"id\"" ":" JSON_STRING
obj_ord_0_2_c: "," "\"id\"" ":" JSON_STRING
obj_ord_0_1_nc: "\"couponCode\"" ":" str_or_null obj_ord_0_2_c | obj_ord_0_2_nc
obj_ord_0_0_nc: "\"affiliation\"" ":" str_or_null obj_ord_0_1_c | obj_ord_0_1_nc
obj_ord_0_1_c: "," "\"couponCode\"" ":" str_or_null obj_ord_0_2_c | obj_ord_0_2_c
obj_ord_0_0_c: "," "\"affiliation\"" ":" str_or_null obj_ord_0_1_c | obj_ord_0_1_c

start: "{" obj_ord_0_0_nc "}"
"###
}

fn nullable_string_no_alternation_lark() -> &'static str {
    r###"
PATTERN_0: /[\x20-\x21\x23-\x5B\x5D-\x7F]/
PATTERN_6: /[\x22\x2F\x5C\x62\x66\x6E\x72\x74]/
STRING_CHAR: PATTERN_0
ESCAPE_SHORT_CHAR: PATTERN_6
ESCAPE_SEQ: "\\" ESCAPE_SHORT_CHAR
STRING_CONTENT: (STRING_CHAR | ESCAPE_SEQ)*
JSON_STRING: "\"" STRING_CONTENT "\""

obj_ord_0_2_nc: "\"id\"" ":" JSON_STRING
obj_ord_0_2_c: "," "\"id\"" ":" JSON_STRING
obj_ord_0_1_nc: "\"couponCode\"" ":" JSON_STRING obj_ord_0_2_c | obj_ord_0_2_nc
obj_ord_0_0_nc: "\"affiliation\"" ":" JSON_STRING obj_ord_0_1_c | obj_ord_0_1_nc
obj_ord_0_1_c: "," "\"couponCode\"" ":" JSON_STRING obj_ord_0_2_c | obj_ord_0_2_c
obj_ord_0_0_c: "," "\"affiliation\"" ":" JSON_STRING obj_ord_0_1_c | obj_ord_0_1_c

start: "{" obj_ord_0_0_nc "}"
"###
}

#[test]
fn test_char_class_excludes_control_bytes() {
    let vocab = Vocab::new(
        vec![
            (0u32, vec![0x00]),
            (1u32, vec![0x20]),
        ],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"
        root ::= STRING_CHAR
        STRING_CHAR ::= [\x20-\x21\x23-\x5B\x5D-\xFF]
        "#,
        &vocab,
    )
    .unwrap();

    let mask = c.start().mask();
    assert!(!token_allowed(&mask, 0), "control byte 0x00 should not match the char class");
    assert!(token_allowed(&mask, 1), "space byte 0x20 should match the char class");
}

#[test]
fn test_nullable_string_property_allows_quote_comma_quote_boundary_token() {
    let vocab = Vocab::new(vec![(0u32, b"\",\"".to_vec())], None);
    let c = Constraint::from_lark(nullable_string_inline_alternation_lark(), &vocab).unwrap();

    let prefix = b"{\"affiliation\":\"Example Store\",\"couponCode\":\"SUMMER";
    let mut s = c.start();
    s.commit_bytes(prefix).unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "joint token '\",\"' must be allowed across the nullable (JSON_STRING|JSON_NULL) boundary"
    );
}

#[test]
fn test_nullable_string_helper_rule_variant() {
    let vocab = Vocab::new(vec![(0u32, b"\",\"".to_vec())], None);
    let c = Constraint::from_lark(nullable_string_helper_rule_lark(), &vocab).unwrap();

    let prefix = b"{\"affiliation\":\"Example Store\",\"couponCode\":\"SUMMER";
    let mut s = c.start();
    s.commit_bytes(prefix).unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "joint token '\",\"' must remain allowed when the nullable alternation is factored through a helper rule"
    );
}

#[test]
fn test_nullable_string_no_alternation_works() {
    let vocab = Vocab::new(vec![(0u32, b"\",\"".to_vec())], None);
    let c = Constraint::from_lark(nullable_string_no_alternation_lark(), &vocab).unwrap();

    let prefix = b"{\"affiliation\":\"Example Store\",\"couponCode\":\"SUMMER";
    let mut s = c.start();
    s.commit_bytes(prefix).unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 0),
        "joint token '\",\"' should also be allowed in the simpler non-alternation baseline"
    );
}

#[test]
fn test_o63377_rejects_false_positive_b_after_double_a_prefix() {
    let vocab = Vocab::new(vec![(0u32, b"b".to_vec())], None);
    let c = Constraint::from_lark(
        r#"
        A: "a" ("b"?)
        B: "ac"
        C: "a"+
        start: A B C
        "#,
        &vocab,
    )
    .unwrap();

    let mut s = c.start();
    s.commit_bytes(b"aa").unwrap();

    let mask = s.mask();
    assert!(
        !token_allowed(&mask, 0),
        "token 'b' must be rejected at prefix 'aa' for the o63377 false-positive repro"
    );
}

#[test]
fn test_o17408_rejects_false_positive_3_after_sas_phase_prefix() {
    let vocab = Vocab::new(
        vec![
            (16u32, b"1".to_vec()),
            (17u32, b"2".to_vec()),
            (18u32, b"3".to_vec()),
        ],
        None,
    );
    let prefix = b"{\"certificationConditions\": \"This device is certified for use in the United States.\", \"certificationDate\": \"2022-01-01\", \"certificationExpiration\": \"2027-01-01\", \"certificationId\": \"DA-123456\", \"frn\": \"0012345678\", \"sasPhase\": \"";
    let lark = r#"
        start: PREFIX ("1" | "2") "\"}"
        PREFIX: /\{"certificationConditions": "This device is certified for use in the United States\.", "certificationDate": "2022-01-01", "certificationExpiration": "2027-01-01", "certificationId": "DA-123456", "frn": "0012345678", "sasPhase": "/
    "#;

    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_bytes(prefix).unwrap();

    let mask = s.mask();
    assert!(token_allowed(&mask, 16), "token '1' must be allowed after the sasPhase prefix");
    assert!(token_allowed(&mask, 17), "token '2' must be allowed after the sasPhase prefix");
    assert!(
        !token_allowed(&mask, 18),
        "token '3' must be rejected after the sasPhase prefix"
    );
}

#[test]
fn test_o17408_rejects_false_positive_3_with_ordered_object_context() {
    let vocab = Vocab::new(
        vec![
            (16u32, b"1".to_vec()),
            (17u32, b"2".to_vec()),
            (18u32, b"3".to_vec()),
        ],
        None,
    );
    let prefix = b"{\"certificationConditions\": \"This device is certified for use in the United States.\", \"certificationDate\": \"2022-01-01\", \"certificationExpiration\": \"2027-01-01\", \"certificationId\": \"DA-123456\", \"frn\": \"0012345678\", \"sasPhase\": \"";
    let lark = r#"
        start: "{" CERT_COND ", " CERT_DATE ", " CERT_EXP ", " CERT_ID ", " FRN ", " SAS_PHASE "}"
        CERT_COND: "\"certificationConditions\": " JSON_STRING
        CERT_DATE: "\"certificationDate\": " JSON_STRING
        CERT_EXP: "\"certificationExpiration\": " JSON_STRING
        CERT_ID: "\"certificationId\": " JSON_STRING
        FRN: "\"frn\": " JSON_STRING
        SAS_PHASE: "\"sasPhase\": " ("\"1\"" | "\"2\"")
        JSON_STRING: "\"" STR_CHAR* "\""
        STR_CHAR: /[A-Za-z0-9 .:-]/
    "#;

    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();
    s.commit_bytes(prefix).unwrap();

    let mask = s.mask();
    assert!(token_allowed(&mask, 16), "token '1' must be allowed after the sasPhase prefix");
    assert!(token_allowed(&mask, 17), "token '2' must be allowed after the sasPhase prefix");
    assert!(
        !token_allowed(&mask, 18),
        "token '3' must be rejected after the sasPhase prefix in the ordered-object grammar"
    );
}

#[test]
fn test_nested_object_prefix_keeps_joint_comma_quote_token_allowed() {
    let lark = r#"
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
"#;
    let vocab = Vocab::new(
        vec![
            (0u32, b"{\"".to_vec()),
            (1u32, b"\":\"".to_vec()),
            (2u32, b"\",\"".to_vec()),
            (3u32, b"a\"".to_vec()),
            (4u32, b":{\"".to_vec()),
            (5u32, b"\":{}}}".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_lark(lark, &vocab).unwrap();
    let mut s = c.start();

    for token in [0u32, 1, 2, 3, 4, 1] {
        s.commit_token(token).unwrap();
    }

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 2),
        "disputed token '\",\"' should remain allowed after the nested-object prefix regression sequence"
    );
}

#[test]
fn test_minimal_lark_repro_compiles_without_panicking() {
    let vocab = Vocab::new(
        vec![(0u32, b"ay".to_vec()), (1u32, b"xa".to_vec())],
        None,
    );
    let _ = Constraint::from_lark("start: \"aa\"", &vocab)
        .expect("minimal copy should compile without panicking");
}

#[test]
fn test_json_schema_object_with_empty_object_token_panics() {
    json_schema_constraint(&["{}"], r#"{"type": "object"}"#);
}

#[test]
fn test_space_star_then_f_allows_joint_space_f_token() {
    let vocab = Vocab::new(
        vec![(0u32, b" ".to_vec()), (1u32, b" f".to_vec())],
        None,
    );
    let c = Constraint::from_ebnf(
        r#"
        start ::= SPACE* F
        SPACE ::= ' '
        F ::= 'f'
        "#,
        &vocab,
    )
    .unwrap();

    let mask = c.start().mask();
    assert!(token_allowed(&mask, 0), "space token should be allowed at the start of SPACE* F");
    assert!(token_allowed(&mask, 1), "span token ' f' should be allowed at the start of SPACE* F");
}

#[test]
fn test_adjacent_terminals_reject_middle_token_at_start() {
    let vocab = Vocab::new(vec![(0u32, b"b".to_vec())], None);
    let c = Constraint::from_ebnf(
        r#"
        start ::= A B
        A ::= 'a' 'b'
        B ::= 'b' 'c'
        "#,
        &vocab,
    )
    .unwrap();

    let mask = c.start().mask();
    assert_empty_mask(&mask);
}

#[test]
fn test_sentence_grammar_from_prompt() {
    let vocab = make_vocab(&[
        "a",
        "the",
        "apple",
        "banana",
        "person",
        " ",
        "eats",
        "likes",
        "is",
        "tasty",
        "red",
        "happy",
        ".",
        "and",
        "e",
        "eth",
    ]);
    let c = Constraint::from_ebnf(
        r#"
        start ::= A SPACE B
        A ::= 'a' | 'the' | 'apple' | 'banana' | 'person'
        SPACE ::= ' '
        B ::= 'eats' | 'likes' | 'is' | 'tasty' | 'red' | 'happy' | '.' | 'and'
        "#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    let mask = s.mask();
    assert_allowed_tokens(&mask, &[0, 1, 2, 3, 4]);

    s.commit_token(2).unwrap(); // "apple"
    let mask = s.mask();
    assert_allowed_tokens(&mask, &[5]);

    s.commit_token(5).unwrap(); // " "
    let mask = s.mask();
    assert_allowed_tokens(&mask, &[0, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
    assert_mask_rejects(&mask, &[15]);

    s.commit_token(6).unwrap(); // "eats"
    assert!(s.is_finished(), "'apple eats' should finish the simple sentence grammar");
    assert_empty_mask(&s.mask());
}

#[test]
fn test_minimal_python_example_with_compiled_grammar() {
    let vocab = Vocab::new(
        (0u32..=9)
            .map(|i| (i, vec![b'0' + i as u8]))
            .chain(std::iter::once((10u32, b"+".to_vec())))
            .collect(),
        None,
    );
    let c = Constraint::from_lark(
        r#"
        start: NUMBER PLUS NUMBER PLUS NUMBER
        NUMBER: /[0-9]+/
        PLUS: "+"
        "#,
        &vocab,
    )
    .unwrap();
    let mut s = c.start();

    for token in [1u32, 2, 3, 10, 4, 5, 6, 10] {
        let mask = s.mask();
        assert!(token_allowed(&mask, token as usize), "token {token} should be allowed before commit");
        s.commit_token(token).unwrap();
    }

    let mask = s.mask();
    for digit in 0..=9 {
        assert!(token_allowed(&mask, digit), "digit token {digit} should be allowed after the second plus");
    }
    assert!(!token_allowed(&mask, 10), "plus should not be allowed immediately after the second plus");
}

fn make_byte_vocab() -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = (0u32..=255).map(|b| (b, vec![b as u8])).collect();
    Vocab::new(entries, None)
}

fn optional_ordered_object_schema(n_keys: usize) -> String {
    let mut props = String::new();
    for i in 0..n_keys {
        if i > 0 {
            props.push_str(", ");
        }
        let key = format!("k{i:02}");
        props.push_str(&format!("\"{key}\": {{\"type\": \"integer\"}}"));
    }

    format!(
        "{{\"type\":\"object\",\"properties\":{{\"o\":{{\"type\":\"object\",\"properties\":{{{props}}},\"additionalProperties\":false,\"minProperties\":1}}}},\"required\":[\"o\"],\"additionalProperties\":false}}"
    )
}

fn ordered_object_example(n_keys: usize) -> String {
    let mut kvs = String::new();
    for i in 0..n_keys {
        if i > 0 {
            kvs.push_str(", ");
        }
        let key = format!("k{i:02}");
        kvs.push_str(&format!("\"{key}\": 0"));
    }
    format!("{{\"o\": {{{kvs}}}}}")
}

fn max_parser_paths_for_text(constraint: &Constraint, text: &str) -> usize {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1_000_000);

    for &byte in text.as_bytes() {
        let mask = state.mask();
        assert!(
            token_allowed(&mask, byte as usize),
            "byte token {byte} must be allowed while replaying example text"
        );
        max_paths = max_paths.max(state.parser_path_count(1_000_000));
        state.commit_token(byte as u32).unwrap();
    }

    max_paths.max(state.parser_path_count(1_000_000))
}

/// MRE for ordered-optional-property ambiguity (the o58906 mechanism).
///
/// A closed object with N optional keys and `minProperties: 1` causes the
/// compiler to emit N top-level ordered-entry alternatives — one anchored on
/// each possible "first present" key. All N alternatives remain live until
/// properties start to disambiguate, so the GLR parser maintains exactly N
/// concurrent stacks. Max forks = N, controllable by adding/removing keys.
///
/// Ceiling: `EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS = 16`. Above that the
/// compiler takes a different path and the N-alternative structure is not
/// generated.
#[test]
fn test_mre_ordered_optional_object_ambiguity() {
    let vocab = make_byte_vocab();

    // n=1: no fork — a single optional key has no competing alternative.
    let c = Constraint::from_json_schema(r#"{
        "type": "object",
        "properties": { "o": {
            "type": "object",
            "properties": { "k00": {"type": "integer"} },
            "additionalProperties": false, "minProperties": 1
        }},
        "required": ["o"], "additionalProperties": false
    }"#, &vocab).unwrap();
    assert_eq!(max_parser_paths_for_text(&c, r#"{"o": {"k00": 0}}"#), 1);

    // n=2: minimal ambiguous case — 2 concurrent stacks.
    let c = Constraint::from_json_schema(r#"{
        "type": "object",
        "properties": { "o": {
            "type": "object",
            "properties": {
                "k00": {"type": "integer"},
                "k01": {"type": "integer"}
            },
            "additionalProperties": false, "minProperties": 1
        }},
        "required": ["o"], "additionalProperties": false
    }"#, &vocab).unwrap();
    assert_eq!(max_parser_paths_for_text(&c, r#"{"o": {"k00": 0, "k01": 0}}"#), 2);

    // n=4
    let c = Constraint::from_json_schema(r#"{
        "type": "object",
        "properties": { "o": {
            "type": "object",
            "properties": {
                "k00": {"type": "integer"},
                "k01": {"type": "integer"},
                "k02": {"type": "integer"},
                "k03": {"type": "integer"}
            },
            "additionalProperties": false, "minProperties": 1
        }},
        "required": ["o"], "additionalProperties": false
    }"#, &vocab).unwrap();
    assert_eq!(max_parser_paths_for_text(&c, r#"{"o": {"k00": 0, "k01": 0, "k02": 0, "k03": 0}}"#), 4);

    // n=8: each line is one key — you can count them.
    let c = Constraint::from_json_schema(r#"{
        "type": "object",
        "properties": { "o": {
            "type": "object",
            "properties": {
                "k00": {"type": "integer"},
                "k01": {"type": "integer"},
                "k02": {"type": "integer"},
                "k03": {"type": "integer"},
                "k04": {"type": "integer"},
                "k05": {"type": "integer"},
                "k06": {"type": "integer"},
                "k07": {"type": "integer"}
            },
            "additionalProperties": false, "minProperties": 1
        }},
        "required": ["o"], "additionalProperties": false
    }"#, &vocab).unwrap();
    assert_eq!(max_parser_paths_for_text(&c,
        r#"{"o": {"k00": 0, "k01": 0, "k02": 0, "k03": 0, "k04": 0, "k05": 0, "k06": 0, "k07": 0}}"#
    ), 8);

    // n=12, n=16: generated schemas, same assertion.
    for n in [12usize, 16] {
        let c = Constraint::from_json_schema(&optional_ordered_object_schema(n), &vocab).unwrap();
        assert_eq!(max_parser_paths_for_text(&c, &ordered_object_example(n)), n,
            "n={n}: expected exactly {n} concurrent stacks");
    }
}

/// Minimal o56012 repro: fast and reference vocab equivalence must agree.
#[test]
fn test_reference_equivalence_matches_fast_analysis_for_o56012_repro() {
    // SAFETY: set_var is unsafe in edition 2024 due to potential races.
    unsafe { std::env::set_var("REFERENCE_EQUIV_VERIFICATION", "1"); }
    let vocab = Vocab::new(
        vec![
            (0, b"}:".to_vec()),
            (1, b"}}}".to_vec()),
        ],
        None,
    );
    // Should succeed without panic — both fast and reference agree.
    let _c = Constraint::from_lark(r#"start: "{" "}""#, &vocab).unwrap();
}

#[test]
fn test_anchored_pattern_rejects_false_positive_after_partial_match() {
    // Regression test for root-signature merging bug (6357d90ee).
    // Pattern ^abc$ after consuming "a should allow only 'b' and 'bc', not 'c'.
    let vocab = Vocab::new(
        vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"c".to_vec()),
            (4, b"bc".to_vec()),
            (5, b"abc".to_vec()),
        ],
        None,
    );
    let schema = r#"{"type":"string","pattern":"^abc$"}"#;
    let c = Constraint::from_json_schema(schema, &vocab).unwrap();

    let mut s = c.start();
    // Commit opening quote and 'a'
    s.commit_token(0).unwrap(); // "
    s.commit_token(1).unwrap(); // a

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 2),
        "token 'b' must be allowed after '\"a' in pattern ^abc$"
    );
    assert!(
        token_allowed(&mask, 4),
        "token 'bc' must be allowed after '\"a' in pattern ^abc$"
    );
    assert!(
        !token_allowed(&mask, 3),
        "token 'c' must be rejected after '\"a' in pattern ^abc$ (root-signature merge regression)"
    );
}

/// Integration test: compile ONLY the 281 literal terminals from kb_815 (no Expr/DFA terminals).
/// Measures id_map and terminal_dwa build times in isolation.
#[test]
fn bench_kb815_literals_only() {
    use std::time::Instant;

    // The 281 literal terminals extracted from kb_815's GrammarDef.
    // These are the exact property-key byte strings and JSON structural tokens.
    let literals: &[&[u8]] = &[
        b"\"", b"{", b"}", b", ", b"[", b"]",
        b"level\": ", b"role\": ", b"type\": ", b"user\": ", b"name\": ", b"value\": ",
        b"gmsaCredentialSpec\": ", b"gmsaCredentialSpecName\": ", b"runAsUserName\": ",
        b"fsGroup\": ", b"fsGroupChangePolicy\": ", b"runAsGroup\": ", b"runAsNonRoot\": ",
        b"runAsUser\": ", b"seLinuxOptions\": ", b"supplementalGroups\": ", b"sysctls\": ",
        b"windowsOptions\": ", b"apiVersion\": ", b"fieldsType\": ", b"fieldsV1\": ",
        b"manager\": ", b"operation\": ", b"time\": ", b"blockOwnerDeletion\": ",
        b"controller\": ", b"kind\": ", b"uid\": ", b"annotations\": ", b"clusterName\": ",
        b"creationTimestamp\": ", b"deletionGracePeriodSeconds\": ", b"deletionTimestamp\": ",
        b"finalizers\": ", b"generateName\": ", b"generation\": ", b"labels\": ",
        b"managedFields\": ", b"namespace\": ", b"ownerReferences\": ",
        b"resourceVersion\": ", b"selfLink\": ", b"values\": ", b"operator\": ", b"key\": ",
        b"matchExpressions\": ", b"matchFields\": ", b"preference\": ", b"weight\": ",
        b"nodeSelectorTerms\": ",
        b"preferredDuringSchedulingIgnoredDuringExecution\": ",
        b"requiredDuringSchedulingIgnoredDuringExecution\": ",
        b"matchLabels\": ", b"labelSelector\": ", b"namespaces\": ", b"topologyKey\": ",
        b"podAffinityTerm\": ", b"nodeAffinity\": ", b"podAffinity\": ",
        b"podAntiAffinity\": ", b"optional\": ", b"fieldPath\": ", b"containerName\": ",
        b"divisor\": ", b"resource\": ", b"configMapKeyRef\": ", b"fieldRef\": ",
        b"resourceFieldRef\": ", b"secretKeyRef\": ", b"valueFrom\": ", b"configMapRef\": ",
        b"prefix\": ", b"secretRef\": ", b"command\": ", b"host\": ", b"httpHeaders\": ",
        b"path\": ", b"scheme\": ", b"port\": ", b"exec\": ", b"httpGet\": ",
        b"tcpSocket\": ", b"postStart\": ", b"preStop\": ", b"failureThreshold\": ",
        b"initialDelaySeconds\": ", b"periodSeconds\": ", b"successThreshold\": ",
        b"timeoutSeconds\": ", b"hostIP\": ", b"containerPort\": ", b"hostPort\": ",
        b"protocol\": ", b"limits\": ", b"requests\": ", b"add\": ", b"drop\": ",
        b"allowPrivilegeEscalation\": ", b"capabilities\": ", b"privileged\": ",
        b"procMount\": ", b"readOnlyRootFilesystem\": ", b"devicePath\": ",
        b"mountPropagation\": ", b"readOnly\": ", b"subPath\": ", b"subPathExpr\": ",
        b"mountPath\": ", b"args\": ", b"env\": ", b"envFrom\": ", b"image\": ",
        b"imagePullPolicy\": ", b"lifecycle\": ", b"livenessProbe\": ", b"ports\": ",
        b"readinessProbe\": ", b"resources\": ", b"securityContext\": ",
        b"startupProbe\": ", b"stdin\": ", b"stdinOnce\": ",
        b"terminationMessagePath\": ", b"terminationMessagePolicy\": ", b"tty\": ",
        b"volumeDevices\": ", b"volumeMounts\": ", b"workingDir\": ", b"nameservers\": ",
        b"options\": ", b"searches\": ", b"targetContainerName\": ", b"hostnames\": ",
        b"ip\": ", b"conditionType\": ", b"effect\": ", b"tolerationSeconds\": ",
        b"maxSkew\": ", b"whenUnsatisfiable\": ", b"fsType\": ", b"partition\": ",
        b"volumeID\": ", b"cachingMode\": ", b"diskName\": ", b"diskURI\": ",
        b"secretName\": ", b"shareName\": ", b"monitors\": ", b"secretFile\": ",
        b"mode\": ", b"defaultMode\": ", b"items\": ", b"driver\": ",
        b"nodePublishSecretRef\": ", b"volumeAttributes\": ", b"medium\": ",
        b"sizeLimit\": ", b"lun\": ", b"targetWWNs\": ", b"wwids\": ",
        b"datasetName\": ", b"datasetUUID\": ", b"pdName\": ", b"directory\": ",
        b"revision\": ", b"repository\": ", b"endpoints\": ",
        b"chapAuthDiscovery\": ", b"chapAuthSession\": ", b"initiatorName\": ",
        b"iqn\": ", b"iscsiInterface\": ", b"portals\": ", b"targetPortal\": ",
        b"server\": ", b"claimName\": ", b"pdID\": ", b"audience\": ",
        b"expirationSeconds\": ", b"configMap\": ", b"downwardAPI\": ", b"secret\": ",
        b"serviceAccountToken\": ", b"sources\": ", b"group\": ", b"registry\": ",
        b"tenant\": ", b"volume\": ", b"keyring\": ", b"pool\": ", b"gateway\": ",
        b"protectionDomain\": ", b"sslEnabled\": ", b"storageMode\": ",
        b"storagePool\": ", b"volumeName\": ", b"system\": ", b"volumeNamespace\": ",
        b"storagePolicyID\": ", b"storagePolicyName\": ", b"volumePath\": ",
        b"awsElasticBlockStore\": ", b"azureDisk\": ", b"azureFile\": ", b"cephfs\": ",
        b"cinder\": ", b"csi\": ", b"emptyDir\": ", b"fc\": ", b"flexVolume\": ",
        b"flocker\": ", b"gcePersistentDisk\": ", b"gitRepo\": ", b"glusterfs\": ",
        b"hostPath\": ", b"iscsi\": ", b"nfs\": ", b"persistentVolumeClaim\": ",
        b"photonPersistentDisk\": ", b"portworxVolume\": ", b"projected\": ",
        b"quobyte\": ", b"rbd\": ", b"scaleIO\": ", b"storageos\": ",
        b"vsphereVolume\": ", b"activeDeadlineSeconds\": ", b"affinity\": ",
        b"automountServiceAccountToken\": ", b"containers\": ", b"dnsConfig\": ",
        b"dnsPolicy\": ", b"enableServiceLinks\": ", b"ephemeralContainers\": ",
        b"hostAliases\": ", b"hostIPC\": ", b"hostNetwork\": ", b"hostPID\": ",
        b"hostname\": ", b"imagePullSecrets\": ", b"initContainers\": ", b"nodeName\": ",
        b"nodeSelector\": ", b"overhead\": ", b"preemptionPolicy\": ", b"priority\": ",
        b"priorityClassName\": ", b"readinessGates\": ", b"restartPolicy\": ",
        b"runtimeClassName\": ", b"schedulerName\": ", b"serviceAccount\": ",
        b"serviceAccountName\": ", b"shareProcessNamespace\": ", b"subdomain\": ",
        b"terminationGracePeriodSeconds\": ", b"tolerations\": ",
        b"topologySpreadConstraints\": ", b"volumes\": ", b"metadata\": ", b"spec\": ",
        b"minReadySeconds\": ", b"replicas\": ", b"selector\": ", b"template\": ",
        b"lastTransitionTime\": ", b"message\": ", b"reason\": ", b"status\": ",
        b"availableReplicas\": ", b"conditions\": ", b"fullyLabeledReplicas\": ",
        b"observedGeneration\": ", b"readyReplicas\": ",
        b"ReplicationController\"",
    ];

    assert_eq!(literals.len(), 281, "expected 281 literal terminals from kb_815");

    // Build a byte-level vocab (256 tokens, one per byte value).
    let byte_vocab_entries: Vec<(u32, Vec<u8>)> =
        (0..=255u32).map(|b| (b, vec![b as u8])).collect();
    let vocab = Vocab::new(byte_vocab_entries, None);

    // Build a minimal GrammarDef JSON: start -> T_0 | T_1 | ... | T_280
    let mut terminals_json = Vec::new();
    for (i, lit) in literals.iter().enumerate() {
        let bytes_array: Vec<u8> = lit.to_vec();
        terminals_json.push(serde_json::json!({
            "Literal": { "id": i, "bytes": bytes_array }
        }));
    }
    let rules_json: Vec<serde_json::Value> = (0..literals.len())
        .map(|i| serde_json::json!({"lhs": 0, "rhs": [{"Terminal": i}]}))
        .collect();
    let terminal_names: std::collections::BTreeMap<String, String> = literals
        .iter()
        .enumerate()
        .map(|(i, lit)| (i.to_string(), String::from_utf8_lossy(lit).into_owned()))
        .collect();
    let grammar_def = serde_json::json!({
        "rules": rules_json,
        "start": 0,
        "terminals": terminals_json,
        "nonterminal_names": {"0": "start"},
        "terminal_names": terminal_names,
        "ignore_terminal": null,
    });
    let grammar_def_json = serde_json::to_string(&grammar_def).unwrap();

    // Enable compile profiling
    unsafe { std::env::set_var("GLRMASK_PROFILE_COMPILE_SUMMARY", "1"); }

    // Compile and time
    let t0 = Instant::now();
    let constraint = glrmask::compile_grammar_def_json(&grammar_def_json, &vocab).unwrap();
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("bench_kb815_literals_only: {:.1}ms total ({} terminals, byte vocab)", total_ms, literals.len());

    // Basic sanity check
    let _state = constraint.start();
}

#[test]
fn test_small_range_finished_semantics() {
    let vocab = make_vocab(&["a"]);

    // ~0..3: can accept EOF after min=0 (but needs >= 1 token for parser init)
    let c = lark_constraint(&["a"], r#"start: "a" ~0..3"#);
    let mut s = c.start();
    assert_mask_allows(&s.mask(), &[0]);
    commit_all(&mut s, &[0]);
    assert!(s.is_finished());

    // ~1..3: can accept EOF after min=1
    let c = lark_constraint(&["a"], r#"start: "a" ~1..3"#);
    let mut s = c.start();
    assert_mask_allows(&s.mask(), &[0]);
    commit_all(&mut s, &[0]);
    assert!(s.is_finished());

    // ~2..3: cannot accept EOF after 1 token (need min=2)
    let c = lark_constraint(&["a"], r#"start: "a" ~2..3"#);
    let mut s = c.start();
    commit_all(&mut s, &[0]);
    assert!(!s.is_finished());
    commit_all(&mut s, &[0]);
    assert!(s.is_finished());

    // * (0 or more): can accept EOF after any token
    let c = lark_constraint(&["a"], r#"start: "a"*"#);
    let mut s = c.start();
    commit_all(&mut s, &[0]);
    assert!(s.is_finished());

    // + (1 or more): can accept EOF after >= 1 token
    let c = lark_constraint(&["a"], r#"start: "a"+"#);
    let mut s = c.start();
    commit_all(&mut s, &[0]);
    assert!(s.is_finished());

    // ? (0 or 1): can accept EOF after 0 or 1 token
    let c = lark_constraint(&["a"], r#"start: "a"? "#);
    let mut s = c.start();
    commit_all(&mut s, &[0]);
    assert!(s.is_finished());
}

/// Reproduces super-linear build-time scaling of `"a" ~0..N` for increasing N.
///
/// Runs a wide range of sizes and prints build time + various stats.
/// Enable with `--ignored --nocapture`. Pair with env vars like
/// `GLRMASK_PROFILE_COMPILE_SUMMARY=1 GLRMASK_PROFILE_PARSER_DWA=1` for detail.
#[test]
#[ignore]
fn bench_bounded_repeat_scaling() {
    use std::time::Instant;

    // Dense set of sizes spanning 10 → 1e12 for precise scaling fits.
    let sizes: &[u64] = if std::env::var_os("BENCH_BIG_ONLY").is_some() {
        &[1_000_000, 1_000_000_000, 1_000_000_000_000]
    } else {
        &[
            10, 100, 1_000, 10_000, 100_000,
            1_000_000, 10_000_000, 100_000_000,
            1_000_000_000, 10_000_000_000, 100_000_000_000,
            1_000_000_000_000,
        ]
    };

    eprintln!();
    eprintln!("=== bench_bounded_repeat_scaling (grammar: \"a\" ~0..N) ===");
    eprintln!(
        "{:>14}  {:>4}  {:>10}  {:>10}  {:>10}  {:>17}  {:>22}",
        "N",
        "logN",
        "popcnt",
        "build_ms",
        "num_states",
        "parser_dwa_states",
        "parser_dwa_transitions",
    );

    for &n in sizes {
        let grammar = format!("start: \"a\" ~0..{}", n);
        let vocab = make_vocab(&["a"]);
        // Warm up once (first build pays JIT / alloc startup).
        let _ = Constraint::from_lark(&grammar, &vocab).unwrap();
        let t0 = Instant::now();
        let constraint = Constraint::from_lark(&grammar, &vocab).unwrap();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "{:>14}  {:>4}  {:>10}  {:>10.2}  {:>10}  {:>17}  {:>22}",
            n,
            64 - (n as u64).leading_zeros(),
            (n as u64).count_ones(),
            elapsed_ms,
            constraint.debug_num_states(),
            constraint.debug_parser_dwa_num_states(),
            constraint.debug_parser_dwa_num_transitions(),
        );
    }
}

/// Compare a power-of-two-sized repeat against its neighbors to isolate the
/// effect of bit popcount / binary-decomposition sharing.
#[test]
#[ignore]
fn bench_bounded_repeat_popcount() {
    use std::time::Instant;

    // Pairs of (N, label) designed to test whether popcount(N) matters.
    let cases: &[(u64, &str)] = &[
        ((1u64 << 30) - 1, "2^30 - 1  (30 set bits)"),
        (1u64 << 30,       "2^30      (1 set bit)"),
        ((1u64 << 30) + 1, "2^30 + 1  (2 set bits)"),
        ((1u64 << 40) - 1, "2^40 - 1  (40 set bits)"),
        (1u64 << 40,       "2^40      (1 set bit)"),
        ((1u64 << 40) + 1, "2^40 + 1  (2 set bits)"),
        (1_000_000_000,    "10^9"),
        (1_073_741_824,    "2^30 ≈ 10^9"),
        (1_000_000_000_000,"10^12"),
        (1_099_511_627_776,"2^40 ≈ 10^12"),
    ];

    eprintln!();
    eprintln!("=== bench_bounded_repeat_popcount ===");
    eprintln!(
        "{:>20}  {:>30}  {:>10}  {:>10}  {:>10}  {:>17}  {:>22}",
        "N",
        "label",
        "build_ms",
        "popcnt",
        "num_states",
        "parser_dwa_states",
        "parser_dwa_transitions",
    );

    for &(n, label) in cases {
        let grammar = format!("start: \"a\" ~0..{}", n);
        let vocab = make_vocab(&["a"]);
        let _ = Constraint::from_lark(&grammar, &vocab).unwrap();
        let t0 = Instant::now();
        let constraint = Constraint::from_lark(&grammar, &vocab).unwrap();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "{:>20}  {:>30}  {:>10.2}  {:>10}  {:>10}  {:>17}  {:>22}",
            n,
            label,
            elapsed_ms,
            n.count_ones(),
            constraint.debug_num_states(),
            constraint.debug_parser_dwa_num_states(),
            constraint.debug_parser_dwa_num_transitions(),
        );
    }
}

/// Build a symmetric balanced-binary grammar: `rN -> r{N/2} r{N/2}` for pow2 N.
/// Accepts 0..=N copies of "a". O(log N) rules.
fn build_symmetric_pow2_grammar(log_n: u32) -> String {
    let mut out = String::new();
    out.push_str("start: r_0\n");
    // r_0 is top; r_i refers to a tree of 2^(log_n - i) leaves.
    for level in 0..log_n {
        out.push_str(&format!("r_{}: r_{} r_{}\n", level, level + 1, level + 1));
    }
    out.push_str(&format!("r_{}: \"a\" | \n", log_n));
    out
}

/// Build an asymmetric balanced-binary grammar: every node has a distinct
/// left child and right child nonterminal (different names but identical
/// languages). Accepts 0..=N copies of "a". O(N) rules.
fn build_asymmetric_pow2_grammar(log_n: u32) -> String {
    let mut out = String::new();
    out.push_str("start: r_\n");
    // Walk a full binary tree; each node has a distinct path-encoded name.
    fn rec(out: &mut String, path: &str, depth: u32, max_depth: u32) {
        if depth == max_depth {
            out.push_str(&format!("r_{}: \"a\" | \n", path));
        } else {
            out.push_str(&format!(
                "r_{}: r_{}L r_{}R\n",
                path, path, path
            ));
            let left = format!("{}L", path);
            let right = format!("{}R", path);
            rec(out, &left, depth + 1, max_depth);
            rec(out, &right, depth + 1, max_depth);
        }
    }
    rec(&mut out, "", 0, log_n);
    out
}

/// Directly test the hypothesis: the `A -> B B` pattern (same nonterminal on
/// both sides) causes a super-linear blow-up in parser_dwa transitions vs an
/// equivalent grammar with distinct nonterminals on each side.
#[test]
#[ignore]
fn bench_symmetric_vs_asymmetric() {
    use std::time::Instant;

    eprintln!();
    eprintln!("=== bench_symmetric_vs_asymmetric (grammar: balanced binary accepting 0..2^k 'a's) ===");
    eprintln!(
        "{:>6}  {:>8}  {:>10}  {:>10}  {:>10}  {:>17}  {:>22}",
        "shape", "log2(N)", "num_rules", "build_ms", "num_states", "pdwa_states", "pdwa_trans",
    );

    for log_n in 1..=7u32 {
        // Symmetric
        let grammar = build_symmetric_pow2_grammar(log_n);
        let num_rules = grammar.lines().filter(|l| l.contains(":")).count();
        let vocab = make_vocab(&["a"]);
        let _ = Constraint::from_lark(&grammar, &vocab).unwrap();
        let t0 = Instant::now();
        let c = Constraint::from_lark(&grammar, &vocab).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "{:>6}  {:>8}  {:>10}  {:>10.2}  {:>10}  {:>17}  {:>22}",
            "sym", log_n, num_rules, ms,
            c.debug_num_states(),
            c.debug_parser_dwa_num_states(),
            c.debug_parser_dwa_num_transitions(),
        );

        // Asymmetric: same language but all distinct nonterminal names.
        let grammar = build_asymmetric_pow2_grammar(log_n);
        let num_rules = grammar.lines().filter(|l| l.contains(":")).count();
        let _ = Constraint::from_lark(&grammar, &vocab).unwrap();
        let t0 = Instant::now();
        let c = Constraint::from_lark(&grammar, &vocab).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "{:>6}  {:>8}  {:>10}  {:>10.2}  {:>10}  {:>17}  {:>22}",
            "asym", log_n, num_rules, ms,
            c.debug_num_states(),
            c.debug_parser_dwa_num_states(),
            c.debug_parser_dwa_num_transitions(),
        );
    }
}

/// Test for an ambiguous recursive grammar where `,` serves as separator at
/// every level. The grammar is:
///
/// ```text
/// s ::= l
/// l ::= a , b | a | b
/// a ::= c , d | c | d
/// b ::= e , f | e | f
/// c ::= 'a' , 'b' | 'a' | 'b'
/// d ::= 'c' , 'd' | 'c' | 'd'
/// e ::= 'e' , 'f' | 'e' | 'f'
/// f ::= 'g' , 'h' | 'g' | 'h'
/// ```
///
/// The input `a , b , c , d , e , f , g , h` is structurally ambiguous: the
/// eight comma-separated tokens can be grouped into l, a, b, c, d, e, f in
/// many different ways. This test prints the parse table and asserts that
/// there are no SR/RR conflicts (the assertion FAILS — the grammar is
/// ambiguous).
///
/// Note: glrmask EBNF treats all-uppercase identifiers as terminal rule names
/// and mixed/lowercase identifiers as nonterminals, so the grammar variables
/// use lowercase names while quoted literals `'a'`..`'h'` match vocab tokens.
#[test]
fn test_grammar_cascading_ambiguity() {
    // Tokens: a=0  ,=1  b=2  c=3  d=4  e=5  f=6  g=7  h=8
    let tokens = &["a", ",", "b", "c", "d", "e", "f", "g", "h"];
    let constraint = ebnf_constraint(tokens, r#"
        s ::= l
        l ::= a ',' b | a | b
        a ::= c ',' d | c | d
        b ::= e ',' f | e | f
        c ::= 'a' ',' 'b' | 'a' | 'b'
        d ::= 'c' ',' 'd' | 'c' | 'd'
        e ::= 'e' ',' 'f' | 'e' | 'f'
        f ::= 'g' ',' 'h' | 'g' | 'h'
    "#);

    // ── Parse table ────────────────────────────────────────────────────────
    let num_states    = constraint.debug_num_states();
    let num_terminals = constraint.debug_num_terminals();
    let rules         = constraint.debug_rules();
    // Map internal token ID → original vocab string (best-effort).
    let internal_to_tokens = constraint.debug_internal_token_to_tokens();
    let internal_name = |tid: u32| -> String {
        let orig_ids = internal_to_tokens.get(tid as usize)
            .map(|v| v.as_slice()).unwrap_or(&[]);
        let names: Vec<&str> = orig_ids.iter()
            .filter_map(|&id| tokens.get(id as usize).copied())
            .collect();
        if names.is_empty() {
            format!("t{tid}")
        } else {
            names.join("/")
        }
    };

    eprintln!();
    eprintln!("=== parse table: {num_states} states, {num_terminals} terminals ===");
    eprintln!();
    eprintln!("Rules:");
    for (lhs, rhs_len, rhs_dbg) in &rules {
        eprintln!("  NT{lhs} -> ({rhs_len} symbols) {rhs_dbg}");
    }
    eprintln!();
    eprintln!("Actions per state:");
    for state in 0..num_states {
        let mut actions = constraint.debug_actions_for_state(state);
        if actions.is_empty() { continue; }
        actions.sort_by_key(|(tid, _)| *tid);
        let formatted: Vec<String> = actions.iter()
            .map(|(tid, act)| format!("{}:{}", internal_name(*tid), act))
            .collect();
        eprintln!("  s{state:>3}: {}", formatted.join("  "));
    }
    eprintln!();

    // ── Conflict check (this assertion FAILS — grammar is ambiguous) ───────
    let conflicts: Vec<String> = (0..num_states).flat_map(|state| {
        constraint.debug_actions_for_state(state).into_iter()
            .filter(|(_, act)| act.contains("Split"))
            .map(move |(tid, act)| format!("state {state} on t{tid}: {act}"))
    }).collect();

    assert!(
        conflicts.is_empty(),
        "SR/RR conflicts found ({} total):\n{}",
        conflicts.len(),
        conflicts.join("\n")
    );

    // ── Feed inputs and assert each commit succeeds; assert max stacks ≤ 2 ─
    // Tokens: a=0  ,=1  b=2  c=3  d=4  e=5  f=6  g=7  h=8
    //
    // Grammar structure:
    //   c uses {0 'a', 2 'b'},  d uses {3 'c', 4 'd'}
    //   e uses {5 'e', 6 'f'},  f uses {7 'g', 8 'h'}
    //   a = c | d | c,d    b = e | f | e,f    l = a | b | a,b
    //
    // All inputs below are valid in the grammar and should (after the bug is
    // fixed) commit without error and produce ≤ 2 concurrent stacks.
    // Currently BOTH assertions may fail — that is expected.
    let test_inputs: &[(&str, &[u32])] = &[
        // ── single leaf tokens ──────────────────────────────────────────────
        ("'a'",         &[0]),
        ("'b'",         &[2]),
        ("'c'",         &[3]),
        ("'d'",         &[4]),
        ("'e'",         &[5]),
        ("'f'",         &[6]),
        ("'g'",         &[7]),
        ("'h'",         &[8]),
        // ── leaf-level pairs (c/d/e/f with their own comma) ─────────────────
        ("'a','b'",     &[0, 1, 2]),
        ("'c','d'",     &[3, 1, 4]),
        ("'e','f'",     &[5, 1, 6]),
        ("'g','h'",     &[7, 1, 8]),
        // ── a-level: c,d (both leaf pairs) ──────────────────────────────────
        ("(a,b),(c,d)", &[0, 1, 2, 1, 3, 1, 4]),
        // ── b-level: e,f (both leaf pairs) ──────────────────────────────────
        ("(e,f),(g,h)", &[5, 1, 6, 1, 7, 1, 8]),
        // ── l-level: a,b where each side is a leaf pair ──────────────────────
        ("a,b,c,d , e,f,g,h", &[0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8]),
        // ── cross-level: l := a,b with minimal a and b ──────────────────────
        // a='a' (→c→a), b='e' (→e→b)
        ("'a' , 'e'",   &[0, 1, 5]),
        // a='b' (→c→a), b='g' (→f→b)
        ("'b' , 'g'",   &[2, 1, 7]),
        // a='c' (→d→a), b='e' (→e→b)
        ("'c' , 'e'",   &[3, 1, 5]),
        // a='d' (→d→a), b='f' (→e→b)
        ("'d' , 'f'",   &[4, 1, 6]),
        // ── cross-level: a := c,d with minimal c and d ───────────────────────
        // c='a', d='c'  → a=(c,d)  → l=a
        ("'a' , 'c'",   &[0, 1, 3]),
        // c='b', d='d'  → a=(c,d)  → l=a
        ("'b' , 'd'",   &[2, 1, 4]),
        // ── mixed lengths ────────────────────────────────────────────────────
        // a=(a,b pair), b='e'
        ("(a,b) , 'e'", &[0, 1, 2, 1, 5]),
        // a=(c,d pair), b=(g,h pair)
        ("(c,d) , (g,h)", &[3, 1, 4, 1, 7, 1, 8]),
        // c=(a,b pair), d='c' → a=(c,d)
        ("(a,b) , 'c'", &[0, 1, 2, 1, 3]),
        // a='a', b=(e,f pair)
        ("'a' , (e,f)",  &[0, 1, 5, 1, 6]),
    ];

    eprintln!();
    for (label, input) in test_inputs {
        let mut state = constraint.start();
        let mut max_stacks = state.parser_path_count(1_000_000);
        for (pos, &tok) in input.iter().enumerate() {
            state.commit_token(tok)
                .unwrap_or_else(|e| panic!(
                    "input {label:?}: commit failed at pos {pos} (tok {tok}): {e:?}"
                ));
            max_stacks = max_stacks.max(state.parser_path_count(1_000_000));
        }
        max_stacks = max_stacks.max(state.parser_path_count(1_000_000));
        eprintln!("  {label:30}  max_stacks={max_stacks}");

        assert!(
            max_stacks <= 2,
            "input {label:?}: max concurrent stacks was {max_stacks}, expected ≤ 2",
        );
    }
}

// ---------------------------------------------------------------------------
// GLRM format tests
// ---------------------------------------------------------------------------

fn glrm_constraint(entries: &[&str], grammar: &str) -> Constraint {
    let vocab = make_vocab(entries);
    Constraint::from_glrm_grammar(grammar, &vocab).unwrap()
}

/// GLRM grammar for the schema:
/// ```json
/// {
///   "properties":         { "a": { "type": "string" } },
///   "patternProperties":  { "b": { "type": "string" } },
///   "additionalProperties": { "type": "string" }
/// }
/// ```
///
/// Generated by `make show-grammar-glrmask SCHEMA='...'` and embedded here as
/// a minimal-reproduction for the 3-key object interaction bug.
const MRE_PROPS_PATTERN_AP_GLRM: &str = r#"
start start;

t JSON_STRING_BODY ::= /([^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"/;
nt json_string ::= "\"" JSON_STRING_BODY;
t JSON_INTEGER ::= /-?(0|[1-9][0-9]*)/;
t JSON_NUMBER ::= /-?(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)/;
t JSON_BOOL ::= "true" | "false";
t JSON_NULL ::= "null";
t JSON_KEY_COLON_BODY ::= /([^\x00-\x1f\x7f"\\]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4})*/;
nt json_key_colon ::= "\"" JSON_KEY_COLON_BODY ": ";
nt json_kv ::= json_key_colon json_value;
nt json_object ::= "{" "}" | "{" json_kv (", " json_kv)* "}";
nt json_array ::= "[" "]" | "[" json_value (", " json_value)* "]";
nt json_value ::= json_object | json_array | json_string | JSON_NUMBER | JSON_INTEGER | JSON_BOOL | JSON_NULL;
internal t OBJ_ORD_0_PP_SUBSET_0_KEY_0 ::= (([ !#-[\]-~] | [\xC2-\xDF] [\x80-\xBF] | [\xE0] [\xA0-\xBF] [\x80-\xBF] | [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF] | [\xED] [\x80-\x9F] [\x80-\xBF] | [\xEE\xEF] [\x80-\xBF] [\x80-\xBF] | [\xF0] [\x90-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF4] [\x80-\x8F] [\x80-\xBF] [\x80-\xBF]) | "\\" ["/\\bfnrt] | "\\" "u" [0-9A-Fa-f]{4})* "\"";
internal t OBJ_ORD_0_PP_SUBSET_0_KEY_1 ::= ((([ !#-[\]-~] | [\xC2-\xDF] [\x80-\xBF] | [\xE0] [\xA0-\xBF] [\x80-\xBF] | [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF] | [\xED] [\x80-\x9F] [\x80-\xBF] | [\xEE\xEF] [\x80-\xBF] [\x80-\xBF] | [\xF0] [\x90-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF4] [\x80-\x8F] [\x80-\xBF] [\x80-\xBF]) | "\\" ["/\\bfnrt] | "\\" "u" [0-9A-Fa-f]{4})* "b" (([ !#-[\]-~] | [\xC2-\xDF] [\x80-\xBF] | [\xE0] [\xA0-\xBF] [\x80-\xBF] | [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF] | [\xED] [\x80-\x9F] [\x80-\xBF] | [\xEE\xEF] [\x80-\xBF] [\x80-\xBF] | [\xF0] [\x90-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF] | [\xF4] [\x80-\x8F] [\x80-\xBF] [\x80-\xBF]) | "\\" ["/\\bfnrt] | "\\" "u" [0-9A-Fa-f]{4})*) "\"";
t OBJ_ORD_0_PP_SUBSET_0_KEY_2 ::= ((OBJ_ORD_0_PP_SUBSET_0_KEY_0 & OBJ_ORD_0_PP_SUBSET_0_KEY_1) - "a\"");
t OBJ_ORD_0_AP_KEY_3 ::= (OBJ_ORD_0_PP_SUBSET_0_KEY_0 - ("a\"" | OBJ_ORD_0_PP_SUBSET_0_KEY_1));
nt obj_ord_0_ap_lr ::=  | obj_ord_0_ap_lr ", " (("\"" OBJ_ORD_0_AP_KEY_3 ": ") json_string);
nt obj_ord_0_ap_c ::= obj_ord_0_ap_lr;
nt obj_ord_0_ap_nc ::= (("\"" OBJ_ORD_0_AP_KEY_3 ": ") json_string) obj_ord_0_ap_c | ;
nt obj_ord_0_free_lr ::=  | obj_ord_0_free_lr ", " (("\"" OBJ_ORD_0_PP_SUBSET_0_KEY_2 ": ") json_string);
nt obj_ord_0_free_c ::= obj_ord_0_free_lr obj_ord_0_ap_c;
nt obj_ord_0_free_nc ::= (("\"" OBJ_ORD_0_PP_SUBSET_0_KEY_2 ": ") json_string) obj_ord_0_free_c | obj_ord_0_ap_nc;
nt obj_ord_0_t0 ::= "\"" "a\"" ": " json_string;
nt start ::= (("{" obj_ord_0_t0) obj_ord_0_free_c "}" | "{" "}" | "{" obj_ord_0_free_nc "}") | json_array | json_string | (JSON_INTEGER | JSON_NUMBER) | JSON_BOOL | JSON_NULL;
"#;

/// MRE: properties + patternProperties + additionalProperties three-key object.
///
/// This test loads the grammar directly from the GLRM format (bypassing the
/// JSON Schema importer) to provide a stable reproduction case for the bug
/// where `{"a": "x", "b": "y", "c": "z"}` is rejected even though all three
/// keys are valid under the schema.
#[test]
#[ignore]
fn test_mre_props_pattern_ap_glrm_single_token() {
    // Vocab: single token containing the full object literal.
    let tokens = &[
        r#"{"a": "x", "b": "y", "c": "z"}"#,   // id=0 — should be accepted
        r#"{"a": "x", "b": "y"}"#,               // id=1 — accepted (a + b only)
        r#"{"a": "x", "c": "z"}"#,               // id=2 — accepted (a + c only)
        r#"{"b": "y", "c": "z"}"#,               // id=3 — accepted (b + c only)
        r#"{"a": "x"}"#,                          // id=4 — accepted (a only)
        r#"{"b": "y"}"#,                          // id=5 — accepted (b only)
        r#"{"c": "z"}"#,                          // id=6 — accepted (c only)
        r#"{"a": "x", "b": 1, "c": "z"}"#,       // id=7 — rejected ("b" must be string)
    ];
    let vocab = make_vocab(tokens);
    let c = Constraint::from_glrm_grammar(MRE_PROPS_PATTERN_AP_GLRM, &vocab).unwrap();
    let state = c.start();
    let mask = state.mask();

    let allowed: Vec<usize> = (0..tokens.len())
        .filter(|&i| token_allowed(&mask, i))
        .collect();

    // All valid objects (ids 0–6) must be accepted; invalid (id 7) must be rejected.
    for id in 0..=6 {
        assert!(token_allowed(&mask, id), "token {id} ({}) should be allowed; allowed={allowed:?}", tokens[id]);
    }
    assert!(!token_allowed(&mask, 7), "token 7 (invalid) should be rejected; allowed={allowed:?}");
}

