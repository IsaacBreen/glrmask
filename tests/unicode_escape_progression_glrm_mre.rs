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

#[test]
fn glrm_unicode_escape_progression_allows_bare_u() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec())], None);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let mut state = constraint.start();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), r#"bare \u should be live because \u00.. is valid"#);
}

#[test]
fn glrm_unicode_escape_progression_rejects_u_c() {
    let vocab = Vocab::new(vec![(0u32, br#"\uC"#.to_vec())], None);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let mut state = constraint.start();

    let mask = state.mask();
    assert!(!token_allowed(&mask, 0), r#"\uC should already be dead for /\\u00.../"#);
}

#[test]
fn glrm_unicode_escape_progression_allows_zero_after_bare_u() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec()), (1u32, b"0".to_vec())], None);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(br#"\u"#).unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, 1), r#"0 should be live after \u for /\\u00.../"#);
}

#[test]
fn glrm_unicode_escape_progression_rejects_c_after_bare_u() {
    let vocab = Vocab::new(vec![(0u32, br#"\u"#.to_vec()), (1u32, b"C".to_vec())], None);
    let constraint = Constraint::from_glrm_grammar(GLRM_UNICODE_ESCAPE_PROGRESS_MRE, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(br#"\u"#).unwrap();

    let mask = state.mask();
    assert!(!token_allowed(&mask, 1), r#"C should be dead after \u for /\\u00.../"#);
}