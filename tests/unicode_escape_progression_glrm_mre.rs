use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

#[test]
fn glrm_unicode_escape_progression_allows_bare_u_but_rejects_u_c() {
    let grammar = r#"
start s;

nt s ::= "\"" esc "\"";
t esc ::= /\\u00(?:[01][0-9A-Fa-f]|7[Ff])/;
"#;

    let quote = 0u32;
    let json_u = 1u32;
    let json_u_c = 2u32;
    let zero = 3u32;
    let upper_c = 4u32;
    let vocab = Vocab::new(
        vec![
            (quote, b"\"".to_vec()),
            (json_u, br#"\u"#.to_vec()),
            (json_u_c, br#"\uC"#.to_vec()),
            (zero, b"0".to_vec()),
            (upper_c, b"C".to_vec()),
        ],
        None,
    );

    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_token(quote).unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, json_u as usize), r#"bare \u should be live because \u00.. is valid"#);
    assert!(!token_allowed(&mask, json_u_c as usize), r#"\uC should already be dead for /\\u00.../"#);

    state.commit_token(json_u).unwrap();
    let post_u = state.mask();
    assert!(token_allowed(&post_u, zero as usize), r#"0 should be live after \u for /\\u00.../"#);
    assert!(!token_allowed(&post_u, upper_c as usize), r#"C should be dead after \u for /\\u00.../"#);
}