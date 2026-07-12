use glrmask::{Constraint, Vocab};

fn byte_vocab() -> Vocab {
    Vocab::new(
        (0u32..=255)
            .map(|byte| (byte, vec![byte as u8]))
            .collect(),
        None,
    )
}

fn constraint(grammar: &str) -> Constraint {
    Constraint::from_glrm_grammar(grammar, &byte_vocab()).unwrap()
}

fn assert_accepts(constraint: &Constraint, bytes: &[u8]) {
    let mut state = constraint.start();
    state.commit_bytes(bytes).unwrap();
    assert!(state.is_finished(), "expected {:?} to finish", String::from_utf8_lossy(bytes));
}

fn assert_rejects(constraint: &Constraint, bytes: &[u8]) {
    let mut state = constraint.start();
    let accepted = state.commit_bytes(bytes).is_ok() && state.is_finished();
    assert!(!accepted, "expected {:?} to be rejected", String::from_utf8_lossy(bytes));
}

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    mask.get(token_id as usize / 32)
        .is_some_and(|word| (word >> (token_id % 32)) & 1 != 0)
}

#[test]
fn named_subgrammar_is_referenced_like_a_nonterminal() {
    let constraint = constraint(
        r#"
start document;

g pair ::= {
    start value;
    nt value ::= "a" "b";
};

nt document ::= "<" pair ">";
"#,
    );

    assert_accepts(&constraint, b"<ab>");
    assert_rejects(&constraint, b"<a>");
}

#[test]
fn subgrammar_without_ignore_does_not_inherit_outer_ignore() {
    let constraint = constraint(
        r#"
start document;
ignore WS;
t WS ::= " "+;

g pair ::= {
    start value;
    nt value ::= "a" "b";
};

nt document ::= "<" pair ">";
"#,
    );

    // Outer ignore is allowed around the subgrammar reference.
    assert_accepts(&constraint, b"  <  ab  >  ");
    // It is not inherited by the subgrammar and therefore cannot occur
    // between the subgrammar's own lexical atoms.
    assert_rejects(&constraint, b"<a b>");
}

#[test]
fn subgrammar_uses_its_own_ignore_and_switches_scopes_at_boundaries() {
    let constraint = constraint(
        r#"
start document;
ignore WS;
t WS ::= " "+;

g pair ::= {
    start value;
    ignore NL;
    t NL ::= "\n"+;
    nt value ::= "a" "b";
};

nt document ::= "<" pair ">";
"#,
    );

    assert_accepts(&constraint, b" <  \n\na\nb\n  > ");
    assert_rejects(&constraint, b"<a b>");
    assert_rejects(&constraint, b"<\n ab>");
    assert_rejects(&constraint, b"<a\nb \n>");
    assert_rejects(&constraint, b"\n<ab>");
}

#[test]
fn ignore_is_allowed_at_start_between_terminals_and_at_end() {
    let constraint = constraint(
        r#"
start value;
ignore WS;
t WS ::= " "+;
nt value ::= "a" "b";
"#,
    );

    assert_accepts(&constraint, b"ab");
    assert_accepts(&constraint, b"   ab");
    assert_accepts(&constraint, b"a   b");
    assert_accepts(&constraint, b"ab   ");
    assert_accepts(&constraint, b"   a   b   ");
}

#[test]
fn ignore_is_not_inserted_inside_a_terminal_match() {
    let constraint = constraint(
        r#"
start value;
ignore WS;
t WS ::= " "+;
t WORD ::= "ab";
nt value ::= WORD;
"#,
    );

    assert_accepts(&constraint, b"  ab  ");
    assert_rejects(&constraint, b"a b");
}

#[test]
fn one_vocab_token_can_cross_outer_ignore_subgrammar_and_inner_ignore_boundaries() {
    let vocab = Vocab::new(
        vec![
            (0, b"<".to_vec()),
            (1, b" \na".to_vec()),
            (2, b"\nb\n  >".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_glrm_grammar(
        r#"
start document;
ignore WS;
t WS ::= " "+;

g pair ::= {
    start value;
    ignore NL;
    t NL ::= "\n"+;
    nt value ::= "a" "b";
};

nt document ::= "<" pair ">";
"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    assert!(token_allowed(&state.mask(), 1));
    state.commit_token(1).unwrap();
    assert!(token_allowed(&state.mask(), 2));
    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
fn subgrammar_ignore_surrounds_special_tokens_without_lexing_their_bytes() {
    let vocab = Vocab::new(
        vec![
            (0, b" ".to_vec()),
            (1, b"<".to_vec()),
            (2, b">".to_vec()),
            (42, b"*".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_glrm_grammar(
        r#"
start document;

g inner ::= {
    start value;
    ignore WS;
    lexer group bytes ::= *;
    t WS ::= " "+;
    nt value ::= @token(42);
};

nt document ::= "<" inner ">";
"#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_bytes(b"< ").unwrap();
    assert!(token_allowed(&state.mask(), 42));
    state.commit_token(42).unwrap();
    state.commit_bytes(b" >").unwrap();
    assert!(state.is_finished());

    let mut bytes_only = constraint.start();
    assert!(bytes_only.commit_bytes(b"< * >").is_err());
}

#[test]
fn subgrammar_cannot_reference_outer_definitions() {
    let error = Constraint::from_glrm_grammar(
        r#"
start document;
t OUTER ::= "a";

g inner ::= {
    start value;
    nt value ::= OUTER;
};

nt document ::= inner;
"#,
        &byte_vocab(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("OUTER"), "{error}");
    assert!(error.contains("scope-local") || error.contains("not visible"), "{error}");
}

#[test]
fn outer_grammar_cannot_reference_subgrammar_private_definitions() {
    let error = Constraint::from_glrm_grammar(
        r#"
start document;

g inner ::= {
    start value;
    nt value ::= "a";
};

nt document ::= value;
"#,
        &byte_vocab(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("value"), "{error}");
    assert!(error.contains("scope-local") || error.contains("not visible"), "{error}");
}

#[test]
fn same_definition_names_can_be_reused_in_different_scopes() {
    let constraint = constraint(
        r#"
start document;
t A ::= "x";

g inner ::= {
    start value;
    t A ::= "a";
    nt value ::= A;
};

nt document ::= A inner;
"#,
    );

    assert_accepts(&constraint, b"xa");
    assert_rejects(&constraint, b"xx");
    assert_rejects(&constraint, b"aa");
}

#[test]
fn nested_subgrammars_are_scope_local_and_composable() {
    let constraint = constraint(
        r#"
start document;

g middle ::= {
    start value;

    g leaf ::= {
        start value;
        nt value ::= "a";
    };

    nt value ::= "[" leaf "]";
};

nt document ::= "<" middle ">";
"#,
    );

    assert_accepts(&constraint, b"<[a]>");
}

#[test]
fn ignore_must_be_local_emitting_nonnullable_and_implicit() {
    let missing = Constraint::from_glrm_grammar(
        r#"
start value;
ignore WS;
nt value ::= "a";
"#,
        &byte_vocab(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing.contains("not defined"), "{missing}");

    let nullable = Constraint::from_glrm_grammar(
        r#"
start value;
ignore WS;
t WS ::= " "*;
nt value ::= "a";
"#,
        &byte_vocab(),
    )
    .unwrap_err()
    .to_string();
    assert!(nullable.contains("consume at least one byte"), "{nullable}");

    let explicit = Constraint::from_glrm_grammar(
        r#"
start value;
ignore WS;
t WS ::= " "+;
nt value ::= WS "a";
"#,
        &byte_vocab(),
    )
    .unwrap_err()
    .to_string();
    assert!(explicit.contains("referenced explicitly"), "{explicit}");
}

#[test]
fn verbose_subgrammar_keyword_remains_an_alias_for_g() {
    let constraint = constraint(
        r#"
start document;
subgrammar inner ::= {
    start value;
    nt value ::= "a";
};
nt document ::= inner;
"#,
    );

    assert_accepts(&constraint, b"a");
}
