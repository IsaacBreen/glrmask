use glrmask::{Constraint, Vocab};

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

#[test]
fn repeated_regex_terminal_preserves_distinct_gss_paths() {
    let grammar = r#"
        start start;
        t item ::= /1+/;
        nt start ::= item+;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab(&["1"])).unwrap();
    let mut state = constraint.start();

    state.commit_token(0).unwrap();

    let stacks = state.debug_parser_stacks();
    assert_eq!(state.parser_path_count(1_000_000), 2, "{stacks:?}");
}
