use crate::grammar::ast::lower;

#[test]
fn direct_regular_metadata_survives_compile_preparation() {
    let mut source = String::from("start: s0\n");
    for index in 0..40 {
        source.push_str(&format!("s{index}: A s{} | B\n", index + 1));
    }
    source.push_str("s40: B\nA: /a/\nB: /b/\n");

    let mut named = crate::import::lark::parse_lark_to_named_uncompressed(&source).unwrap();
    assert!(crate::grammar::right_linear::compress_large_right_linear_grammar(&mut named));
    let factored = crate::grammar::factoring::factor_named_grammar(named);
    let lowered = crate::grammar::ast::lower(&factored).unwrap();
    assert!(
        lowered.direct_regular_automaton.is_some(),
        "AST lower lost direct regular metadata"
    );
    let prepared =
        crate::compiler::grammar::transforms::prepare_grammar_transforms_only(lowered);
    assert!(
        prepared.direct_regular_automaton.is_some(),
        "grammar transforms lost direct regular metadata"
    );
    let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&prepared);
    assert!(
        analyzed.direct_regular_automaton.is_some(),
        "analysis lost direct regular metadata"
    );
}

#[test]
fn lexer_groups_round_trip_and_control_tokenizer_partitions() {
    let grammar = crate::grammar::glrm::from_glrm(
        r#"
start s;
lexer group words ::= A, B;
t A ::= "a";
t B ::= "ab";
t C ::= "z";
nt s ::= A | B | C;
"#,
    )
    .unwrap();
    assert_eq!(
        grammar.lexer_partitions.get("A").map(String::as_str),
        Some("words")
    );
    assert_eq!(
        grammar.lexer_partitions.get("B").map(String::as_str),
        Some("words")
    );
    assert!(!grammar.lexer_partitions.contains_key("C"));

    let dumped = crate::grammar::glrm::to_glrm(&grammar);
    assert!(dumped.contains("lexer group words ::= A, B;"), "{dumped}");
    let reparsed = crate::grammar::glrm::from_glrm(&dumped).unwrap();
    assert_eq!(reparsed.lexer_partitions, grammar.lexer_partitions);

    let lowered = lower(&grammar).unwrap();
    assert_eq!(lowered.lexer_partitions.len(), 2);
    let tokenizer = crate::compiler::pipeline::build_tokenizer_with_partition_options(
        &lowered,
        false,
        false,
    );
    assert!(tokenizer.has_epsilon_transitions());
    assert_eq!(
        tokenizer.initial_epsilon_branch_count(),
        2,
        "A/B should share one component while unspecified C is isolated in stress mode",
    );

}
