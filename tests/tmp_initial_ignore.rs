use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

#[test]
fn tmp_initial_ignore_token() {
    let vocab = Vocab::new(
        vec![
            (0, b" ".to_vec()),
            (1, b"a".to_vec()),
            (2, b" a".to_vec()),
        ],
        None,
    );
    let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+ ;
nt start ::= 'a' ;
"#;

    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut separate = constraint.start();
    assert!(token_allowed(&separate.mask(), 0), "initial space missing from mask");
    separate.commit_token(0).unwrap();
    assert!(token_allowed(&separate.mask(), 1), "a missing after space commit");
    separate.commit_token(1).unwrap();

    let mut combined = constraint.start();
    assert!(token_allowed(&combined.mask(), 2), "combined space+a missing");
    combined.commit_token(2).unwrap();
}
