use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

const GLRM_UNICODE_ESCAPE_PROGRESS_MRE: &str = r#"
start s;

nt s ::= esc "\"";
t esc ::= /\\u00(?:[01][0-9A-Fa-f]|7[Ff])/;
"#;

const GLRM_UNICODE_ESCAPE_PROGRESS_STRUCTURED_MRE: &str = r#"
start s;

nt s ::= esc "\"";
nt esc ::= "\\u" "0" "0" tail;
t tail ::= /(?:[01][0-9A-Fa-f]|7[Ff])/;
"#;

const GLRM_TRAILING_BACKSLASH_ONE_TOKEN_MRE: &str = r#"
start s;

nt s ::= " " "\"" esc;
t esc ::= /\\u0/;
"#;

#[test]
fn glrm_unicode_escape_progression_allows_bare_u() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let state = constraint.start();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), r#"bare \u should be live because \u00.. is valid"#);
}

#[test]
fn glrm_unicode_escape_progression_rejects_u_c() {
    let vocab = Vocab::new(vec![(0u32, br#"\uC"#.to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let state = constraint.start();

    let mask = state.mask();
    assert!(!token_allowed(&mask, 0), r#"\uC should already be dead for /\\u00.../"#);
}

#[test]
fn glrm_unicode_escape_progression_accepts_bare_u_without_full_vocab_continuation() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec()), (1u32, b"0".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let mut state = constraint.start();

    state
        .commit_bytes(br#"\u"#)
        .expect(r#"byte commits may leave a live language prefix even without a complete vocab continuation"#);
}

#[test]
fn glrm_unicode_escape_progression_accepts_bare_u_with_only_dead_continuation() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec()), (1u32, b"C".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let mut state = constraint.start();

    state
        .commit_bytes(br#"\u"#)
        .expect(r#"byte commits may leave a live language prefix even when current vocab continuations are dead"#);
}

#[test]
fn structured_glrm_unicode_escape_progression_rejects_bare_backslash() {
    let vocab = Vocab::new(vec![(0u32, br#"\\"#.to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_STRUCTURED_MRE, &vocab).unwrap();
    let state = constraint.start();

    let mask = state.mask();
    assert!(
        !token_allowed(&mask, 0),
        r#"structured literal tokens require the full \u prefix rather than a bare backslash token"#
    );
}

#[test]
fn structured_glrm_unicode_escape_progression_allows_bare_u() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_STRUCTURED_MRE, &vocab).unwrap();
    let state = constraint.start();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), r#"bare \u should stay live in structured GLRM"#);
}

#[test]
fn structured_glrm_trailing_backslash_byte_prefix_is_accepted() {
    let vocab = Vocab::new(vec![(0u32, b" \"\\".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(GLRM_TRAILING_BACKSLASH_ONE_TOKEN_MRE, &vocab).unwrap();
    let mut state = constraint.start();

    state
        .commit_bytes(b" \"\\")
        .expect(r#"byte commits may leave a trailing partial escape prefix"#);
}
