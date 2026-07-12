use glrmask::{Constraint, Vocab};

#[test]
fn ti_separator_transport_matches_ti_off_reference() {
    unsafe {
        std::env::set_var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE", "1");
        std::env::set_var("GLRMASK_FORCE_ALL_L2P", "1");
        std::env::set_var("GLRMASK_SPLIT_L2P_VOCAB", "0");
    }

    let grammar = r#"
start S;
t ITEM ::= ", ";
t KEY ::= ": ";
t BOOL ::= "true";
t X ::= "x";
nt S ::= X KEY X | X ITEM BOOL | BOOL KEY BOOL | BOOL ITEM X;
"#;
    let vocab = Vocab::new(vec![(0, b" t".to_vec())], None);
    Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
}
