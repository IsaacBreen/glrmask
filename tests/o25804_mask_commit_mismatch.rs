use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    word < mask.len() && ((mask[word] >> bit) & 1) != 0
}


#[test]
fn max_length_string_quote_token_can_be_committed_even_when_mask_omits_it() {
    const C_TOKEN_ID: u32 = 0;
    const C_TOKEN_BYTES: &[u8] = b"c";
    const QUOTE_TOKEN_ID: u32 = 1;
    const QUOTE_TOKEN_BYTES: &[u8] = b"\"";
    const SCHEMA: &str = r#"{
      "type": "object",
      "properties": {
        "a": {
          "type": "string",
          "maxLength": 2
        }
      },
      "required": ["a"],
      "additionalProperties": false
    }"#;
    const PREFIX: &[u8] = br#"{"a": "x"#;

    fn minimal_vocab() -> Vocab {
        Vocab::new(
            vec![
                (C_TOKEN_ID, C_TOKEN_BYTES.to_vec()),
                (QUOTE_TOKEN_ID, QUOTE_TOKEN_BYTES.to_vec()),
            ],
            None,
        )
    }

    let constraint = Constraint::from_json_schema(SCHEMA, &minimal_vocab()).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX).unwrap();

    let mask_has_quote = token_allowed(&base.mask(), QUOTE_TOKEN_ID);
    dbg!(mask_has_quote);

    let mut commit_bytes_state = base.clone();
    let commit_bytes_ok = commit_bytes_state.commit_bytes(QUOTE_TOKEN_BYTES).is_ok();
    dbg!(commit_bytes_ok);

    let mut commit_token_state = base.clone();
    let commit_token_ok = commit_token_state.commit_token(QUOTE_TOKEN_ID).is_ok();
    dbg!(commit_token_ok);

    assert!(
        !mask_has_quote && commit_token_ok && commit_bytes_ok,
        "expected crate-level mask/commit mismatch, got mask_has_quote={mask_has_quote}, commit_token_ok={commit_token_ok}, commit_bytes_ok={commit_bytes_ok}, stacks={:?}",
        base.debug_parser_stacks(),
    );
}

#[test]
fn max_length_string_quote_token_can_be_committed_even_when_mask_omits_it_glrm() {
    const A_TOKEN_ID: u32 = 0;
    const A_TOKEN_BYTES: &[u8] = b"a";
    const EOS_TOKEN_ID: u32 = 1;
    const EOS_TOKEN_BYTES: &[u8] = b"$";
    const GLRM: &str = r#"
        start start;

        nt start ::= a_rep "$";
        t a_rep ::= "a"{1,2};
    "#;
    const PREFIX: &[u8] = br#"a"#;

    fn minimal_vocab() -> Vocab {
        Vocab::new(
            vec![
                (A_TOKEN_ID, A_TOKEN_BYTES.to_vec()),
                (EOS_TOKEN_ID, EOS_TOKEN_BYTES.to_vec()),
            ],
            None,
        )
    }

    let constraint = Constraint::from_glrm_grammar(GLRM, &minimal_vocab()).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX).unwrap();

    let mask_has_quote = token_allowed(&base.mask(), EOS_TOKEN_ID);
    dbg!(mask_has_quote);

    let mut commit_bytes_state = base.clone();
    let commit_bytes_ok = commit_bytes_state.commit_bytes(EOS_TOKEN_BYTES).is_ok();
    dbg!(commit_bytes_ok);

    let mut commit_token_state = base.clone();
    let commit_token_ok = commit_token_state.commit_token(EOS_TOKEN_ID).is_ok();
    dbg!(commit_token_ok);

    assert!(
        !mask_has_quote && commit_token_ok && commit_bytes_ok,
        "expected crate-level mask/commit mismatch, got mask_has_quote={mask_has_quote}, commit_token_ok={commit_token_ok}, commit_bytes_ok={commit_bytes_ok}, stacks={:?}",
        base.debug_parser_stacks(),
    );
}

#[test]
fn max_length_string_quote_token_can_be_committed_even_when_mask_omits_it_glrm_ids_swapped() {
    const A_TOKEN_ID: u32 = 1;
    const A_TOKEN_BYTES: &[u8] = b"a";
    const EOS_TOKEN_ID: u32 = 0;
    const EOS_TOKEN_BYTES: &[u8] = b"$";
    const GLRM: &str = r#"
        start start;

        nt start ::= a_rep "$";
        t a_rep ::= "a"{1,2};
    "#;
    const PREFIX: &[u8] = br#"a"#;

    fn minimal_vocab() -> Vocab {
        Vocab::new(
            vec![
                (A_TOKEN_ID, A_TOKEN_BYTES.to_vec()),
                (EOS_TOKEN_ID, EOS_TOKEN_BYTES.to_vec()),
            ],
            None,
        )
    }

    let constraint = Constraint::from_glrm_grammar(GLRM, &minimal_vocab()).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX).unwrap();

    let mask_has_quote = token_allowed(&base.mask(), EOS_TOKEN_ID);
    dbg!(mask_has_quote);

    let mut commit_bytes_state = base.clone();
    let commit_bytes_ok = commit_bytes_state.commit_bytes(EOS_TOKEN_BYTES).is_ok();
    dbg!(commit_bytes_ok);

    let mut commit_token_state = base.clone();
    let commit_token_ok = commit_token_state.commit_token(EOS_TOKEN_ID).is_ok();
    dbg!(commit_token_ok);

    assert!(
        mask_has_quote && commit_token_ok && commit_bytes_ok,
        "expected crate-level mask/commit mismatch, got mask_has_quote={mask_has_quote}, commit_token_ok={commit_token_ok}, commit_bytes_ok={commit_bytes_ok}, stacks={:?}",
        base.debug_parser_stacks(),
    );
}