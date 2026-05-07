//! End-to-end smoke tests for grammar/schema construction, masks, commits, and
//! serialization. Narrow regressions live in dedicated test files.

use glrmask::{Constraint, ConstraintState, Vocab};

fn vocab(entries: &[&str]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
            .collect(),
        None,
    )
}

fn bytes_vocab() -> Vocab {
    Vocab::new((0u8..=255).map(|b| (b as u32, vec![b])).collect(), None)
}

fn ebnf(entries: &[&str], grammar: &str) -> Constraint {
    Constraint::from_ebnf(grammar, &vocab(entries)).unwrap()
}

fn lark(entries: &[&str], grammar: &str) -> Constraint {
    Constraint::from_lark(grammar, &vocab(entries)).unwrap()
}

fn schema(entries: &[&str], schema: &str) -> Constraint {
    Constraint::from_json_schema(schema, &vocab(entries)).unwrap()
}

fn byte_schema(schema: &str) -> Constraint {
    Constraint::from_json_schema(schema, &bytes_vocab()).unwrap()
}

fn allowed(mask: &[u32]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .flat_map(|(word, &bits)| {
            (0..32).filter_map(move |bit| {
                ((bits >> bit) & 1 != 0).then_some(word * 32 + bit as usize)
            })
        })
        .collect()
}

fn assert_allowed(state: &ConstraintState<'_>, expected: &[usize]) {
    assert_eq!(allowed(&state.mask()), expected);
}

fn commit_tokens(state: &mut ConstraintState<'_>, tokens: &[u32]) {
    for &token in tokens {
        state.commit_token(token).unwrap();
    }
}

fn assert_accepts_tokens(constraint: &Constraint, tokens: &[u32]) {
    let mut state = constraint.start();
    commit_tokens(&mut state, tokens);
    assert!(state.is_finished());
}

fn assert_rejects_token(constraint: &Constraint, prefix: &[u32], token: u32) {
    let mut state = constraint.start();
    commit_tokens(&mut state, prefix);
    assert!(state.commit_token(token).is_err());
}

fn max_paths_and_stacks(constraint: &Constraint, text: &str) -> (usize, usize) {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1_000_000);
    let mut max_stacks = stack_count(&state);

    for &byte in text.as_bytes() {
        state.commit_bytes(&[byte]).unwrap();
        max_paths = max_paths.max(state.parser_path_count(1_000_000));
        max_stacks = max_stacks.max(stack_count(&state));
    }

    (max_paths, max_stacks)
}

fn stack_count(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

#[test]
fn ebnf_masks_and_commits() {
    let constraint = ebnf(&["a", "b", "c"], r#"start ::= "a" ("b" | "c")"#);
    let mut state = constraint.start();
    assert_allowed(&state, &[0]);

    state.commit_token(0).unwrap();
    assert_allowed(&state, &[1, 2]);

    state.commit_token(2).unwrap();
    assert!(state.is_finished());
    assert_rejects_token(&constraint, &[0], 0);
}

#[test]
fn ebnf_repetition_and_optional_separator() {
    let constraint = ebnf(
        &["x", ",", ";"],
        r#"start ::= "x" ("," "x")* ";"?"#,
    );

    assert_accepts_tokens(&constraint, &[0]);
    assert_accepts_tokens(&constraint, &[0, 1, 0, 1, 0]);
    assert_accepts_tokens(&constraint, &[0, 1, 0, 2]);
    assert_rejects_token(&constraint, &[0, 1], 2);
}

#[test]
fn lark_literals_choices_and_terminals() {
    let constraint = lark(
        &["a", "b", "."],
        r#"
        start: ITEM "."
        ITEM: "a" | "b"
        "#,
    );

    let mut state = constraint.start();
    assert_allowed(&state, &[0, 1]);
    state.commit_token(1).unwrap();
    assert_allowed(&state, &[2]);
    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
fn lark_rejects_parser_refs_inside_terminals() {
    let result = Constraint::from_lark(
        r#"
        start: A
        A: inner
        inner: "a"
        "#,
        &vocab(&["a"]),
    );
    assert!(result.is_err());
}

#[test]
fn json_schema_scalar_and_enum() {
    let scalar = schema(&["true", "false"], r#"{"type":"boolean"}"#);
    assert_accepts_tokens(&scalar, &[0]);
    assert_accepts_tokens(&scalar, &[1]);

    let enum_schema = schema(&[r#""red""#, r#""blue""#, r#""green""#], r#"{"enum":["red","blue"]}"#);
    assert_accepts_tokens(&enum_schema, &[0]);
    assert_accepts_tokens(&enum_schema, &[1]);
    assert_rejects_token(&enum_schema, &[], 2);
}

#[test]
fn json_schema_rejects_invalid_utf8_in_string() {
    let constraint = byte_schema(r#"{"type":"string"}"#);
    let mut state = constraint.start();
    state.commit_bytes(&[b'"']).unwrap();
    assert!(state.commit_bytes(&[0xff]).is_err());
}

#[test]
fn json_schema_uri_format_accepts_basic_uri() {
    let constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    let mut state = constraint.start();
    state.commit_bytes(br#""https://example.com""#).unwrap();
    assert!(state.is_finished());
}

#[test]
fn nullable_repeat_alternative_accepts_nonempty_branch_before_nullable_suffix() {
    let grammar = r#"
        start s;

        nt s ::= "\"" host "b"* "\"";
        nt host ::= "1" | "a"*;
    "#;

    let tiny_vocab = vocab(&["\"", "a"]);
    let constraint = Constraint::from_glrm_grammar(grammar, &tiny_vocab).unwrap();

    let mut empty_host = constraint.start();
    empty_host.commit_token(0).unwrap();
    empty_host.commit_token(0).unwrap();
    assert!(empty_host.is_finished());

    let mut single_alpha_host = constraint.start();
    single_alpha_host.commit_token(0).unwrap();
    single_alpha_host.commit_token(1).unwrap();
    single_alpha_host.commit_token(0).unwrap();
    assert!(single_alpha_host.is_finished());
}

#[test]
fn explicit_left_recursive_lowered_form_accepts_nonempty_branch_before_suffix() {
    let grammar = r#"
        start s;

        nt s ::= "\"" host bs "\"";
        nt host ::= "1" | as;
        nt as ::= | as "a";
        nt bs ::= | bs "b";
    "#;

    let tiny_vocab = vocab(&["\"", "a"]);
    let constraint = Constraint::from_glrm_grammar(grammar, &tiny_vocab).unwrap();

    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(0).unwrap();
    assert!(state.is_finished());
}

#[test]
fn commit_bytes_and_commit_tokens_agree() {
    let constraint = ebnf(&["a", "b", "ab"], r#"start ::= "a" "b" | "ab""#);

    let mut by_tokens = constraint.start();
    by_tokens.commit_tokens(&[0, 1]).unwrap();
    assert!(by_tokens.is_finished());

    let mut by_bytes = constraint.start();
    by_bytes.commit_bytes(b"ab").unwrap();
    assert!(by_bytes.is_finished());
}

#[test]
fn force_reports_deterministic_prefix() {
    let constraint = ebnf(&["a", "b", "c"], r#"start ::= "a" "b" ("c")?"#);
    let mut state = constraint.start();
    assert_eq!(state.force(), vec![0, 1, 2]);

    state.commit_tokens(&[0, 1]).unwrap();
    assert!(state.is_finished());
}

#[test]
fn save_load_roundtrip_preserves_behavior() {
    let constraint = ebnf(&["a", "b"], r#"start ::= "a" "b""#);
    let bytes = constraint.save();
    let loaded = Constraint::load(&bytes).unwrap();
    assert_accepts_tokens(&loaded, &[0, 1]);
}

#[test]
fn plan_style_mask_buffer_matches_mask() {
    let constraint = ebnf(&["a", "b"], r#"start ::= "a" "b""#);
    let mut state = constraint.start();
    let mut buffer = vec![0; constraint.mask_len()];

    state.fill_mask(&mut buffer);
    assert_eq!(buffer, state.mask());
    assert_allowed(&state, &[0]);

    state.commit_token(0).unwrap();
    state.fill_mask(&mut buffer);
    assert_eq!(buffer, state.mask());
    assert_allowed(&state, &[1]);
}

#[test]
fn direct_glrm_ordered_suffix_model_has_stack_ambiguity() {
    let grammar = r#"
        start start;
        nt f0 ::= "a";
        nt f1 ::= "b";
        nt f2 ::= "c";
        nt f3 ::= "d";
        nt f4 ::= "e";
        nt f5 ::= "f";
        nt f6 ::= "g";
        nt f7 ::= "h";
        nt f8 ::= "i";
        nt v0 ::= f0 "," f1 ("," f2)? "," f3 "," f4 "," f5 "," f6 ("," f7)? ("," f8)?;
        nt v1 ::= f0 "," f1 ("," f2)? "," f3 "," f4 "," f5 ("," f6)? "," f7 ("," f8)?;
        nt v2 ::= f0 "," f1 ("," f2)? "," f3 "," f4 "," f5 ("," f6)? ("," f7)? "," f8;
        nt start ::= v0 | v1 | v2;
    "#;

    let constraint = Constraint::from_glrm_grammar(grammar, &bytes_vocab()).unwrap();
    let (max_paths, max_stacks) = max_paths_and_stacks(&constraint, "a,b,c,d,e,f,g,h");
    assert_eq!((max_paths, max_stacks), (3, 3));
}


#[test]
fn json_schema_kubernetes_container_ports_prefix_has_schema_shaped_two_stack_split() {
    // Minimized from Kubernetes kb_996: this keeps the same two-stack
    // ordered-object/additional-property split shape at an open exact key.
    // The empty property names are intentional minimization artifacts; the
    // required shape is one array property with an item schema ending in
    // open key `g`, plus a later sibling array whose item schema shares the
    // prefix but lacks `g`.
    const K8S_ORDERED_PORTS_SCHEMA_FRAGMENT: &str = r####"
    {
      "properties": {
        "a": {"items": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}, "z": {"type": "string"}}}},
        "b": {"items": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}}
      },
      "additionalProperties": false
    }"####;
    const K8S_ORDERED_PORTS_PREFIX: &[u8] = br####"{"a": [{"x": "", ""####;

    let constraint = Constraint::from_json_schema(K8S_ORDERED_PORTS_SCHEMA_FRAGMENT, &bytes_vocab()).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(K8S_ORDERED_PORTS_PREFIX).unwrap();

    let stacks = state.debug_parser_stacks();
    assert_eq!(state.parser_path_count(1_000_000), 2, "{stacks:?}");
    assert_eq!(stacks.len(), 1, "{stacks:?}");
    assert_eq!(stack_count(&state), 2, "{stacks:?}");

    let stack_values = stacks[0]
        .1
        .iter()
        .map(|(stack, _)| stack.clone())
        .collect::<Vec<_>>();
    assert_eq!(stack_values.len(), 2, "{stacks:?}");

    let shared_prefix_len = stack_values[0]
        .iter()
        .zip(&stack_values[1])
        .take_while(|(left, right)| left == right)
        .count();
    assert!(shared_prefix_len >= 2, "{stacks:?}");
    let mut suffix_lengths = [
        stack_values[0].len() - shared_prefix_len,
        stack_values[1].len() - shared_prefix_len,
    ];
    suffix_lengths.sort_unstable();
    assert_eq!(suffix_lengths, [1, 2], "{stacks:?}");
}

#[test]
fn direct_glrm_minimized_lowered_schema_has_two_stack_split() {
    let grammar = r#"start s;nt k::="a""b"*;nt i::=k"b"?;nt s::="d"i;"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &bytes_vocab()).unwrap();

    let mut state = constraint.start();
    for &byte in b"dab" {
        state.commit_bytes(&[byte]).unwrap();
    }
    let stacks = state.debug_parser_stacks();
    let path_count = state.parser_path_count(10);

    assert_eq!(path_count, 2, "{stacks:?}");
    assert_eq!(stacks.len(), 1, "{stacks:?}");
    assert_eq!(stack_count(&state), 2, "{stacks:?}");

    // This is the minimized GLRM lowering of the JSON-schema split above.
    // `k` models the ordered known-property continuation, while `i` adds the
    // following tail continuation. Both nonterminals are load-bearing for the
    // schema-shaped suffix lengths.
    let stack_values = stacks
        .iter()
        .flat_map(|(_, stacks)| stacks.iter().map(|(stack, _)| stack.clone()))
        .collect::<Vec<_>>();
    assert_eq!(stack_values.len(), 2, "{stacks:?}");
    assert_eq!(stack_values[0][..2], stack_values[1][..2], "{stacks:?}");
    let mut suffix_lengths = [stack_values[0].len() - 2, stack_values[1].len() - 2];
    suffix_lengths.sort_unstable();
    assert_eq!(suffix_lengths, [1, 2], "{stacks:?}");
}

#[test]
fn direct_glrm_minimized_lowered_schema_collapses_when_tail_token_differs() {
    let grammar = r#"start s;nt k::="a""b"*;nt i::=k"c"?;nt s::="d"i;"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &bytes_vocab()).unwrap();

    let mut state = constraint.start();
    for &byte in b"dab" {
        state.commit_bytes(&[byte]).unwrap();
    }
    let stacks = state.debug_parser_stacks();

    // This differs from the split test by one literal: the tail continuation is
    // `"c"?` instead of `"b"?`. The consumed `b` can only be part of `k`'s
    // `"b"*`, so the parser has no competing "known-list vs tail" continuation.
    assert_eq!(state.parser_path_count(10), 1, "{stacks:?}");
    assert_eq!(stacks.len(), 1, "{stacks:?}");
    assert_eq!(stack_count(&state), 1, "{stacks:?}");
}
