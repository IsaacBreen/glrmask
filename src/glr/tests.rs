use crate::glr::grammar::{nt, prod, t, regex_name, NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::{GLRParser, GLRParserState};
use crate::glr::table::{generate_glr_parser, TerminalID};
use crate::glr::analyze::{self, remove_productions_with_undefined_nonterminals, filter_productions_by_reachability, simplify_grammar, resolve_right_recursion}; // Import the analyze module
use crate::glr::stats;
use bimap::BiBTreeMap;
use std::collections::BTreeSet;
use crate::interface::display_productions;
// --- Helper Functions for Tests ---

fn create_simple_parser() -> GLRParser {
    // This grammar is left-recursive but does NOT have length-1 cycles.
    let productions = vec![
        prod("S", vec![nt("A"), t("$")]), // Start rule
        prod("A", vec![nt("A"), t("a")]),
        prod("A", vec![t("b")]),
    ];

    generate_glr_parser(&productions, 0, None)
}

// 4. Validation Scope: The `analyze::validate` function currently checks for missing non-terminals
//    and length-1 cycles. It doesn't detect all potential issues like useless rules (unreachable
//    or non-productive non-terminals), which could be considered a limitation of the validation step.

#[test]
fn test_repetition_no_eof() {
    // Grammar: S -> S 'a' | 'a'
    // This grammar defines a sequence of one or more 'a's.
    // We will parse inputs without a final EOF token and check the state.
    let productions = vec![
        prod("S", vec![nt("S"), t("a")]), // Start rule is recursive
        prod("S", vec![t("a")]),
    ];

    // The start production is the first one, index 0.
    // The parser generation will use EOF as the lookahead for the augmented rule S' -> S,
    // but the grammar itself and the inputs we test will not use an EOF/dollar token.
    let parser = generate_glr_parser(&productions, 0, None);
    println!("Parser: {}", parser);

    let a_token = *parser.terminal_map.get_by_left(&regex_name("a")).unwrap();

    // Test case 1: "a"
    let tokens1 = vec![a_token];
    let mut state1 = parser.init_glr_parser(None);
    state1.parse(&tokens1);
    // After parsing "a", the parser should be in a valid state, having recognized an 'S'.
    // The active state should contain stacks that have successfully parsed 'S'.
    assert!(state1.is_ok(), "Parse should be ok after 'a'");

    // Test case 2: "aaa"
    let tokens2 = vec![a_token, a_token, a_token];
    let mut state2 = parser.init_glr_parser(None);
    state2.parse(&tokens2);
    assert!(state2.is_ok(), "Parse should be ok after 'aaa'");

    // Test case 3: "" (empty)
    let tokens3 = vec![];
    let mut state3 = parser.init_glr_parser(None);
    state3.parse(&tokens3);
    // The initial state is ok because it's ready to parse.
    // After parsing nothing, it's still in that same ready state.
    // However, if we were to check for *acceptance*, it would fail because S is not nullable.
    // The `is_ok` check is about whether the parser can continue, not if it has accepted.
    assert!(state3.is_ok(), "State should still be ok after parsing empty input");

    // Test case 4: "b" (invalid token)
    // We need to add 'b' to the grammar to get a token ID for it, but ensure it's not part of the main language.
    let productions_with_b = vec![
        prod("S", vec![nt("S"), t("a")]),
        prod("S", vec![t("a")]),
        prod("Other", vec![t("b")]), // Another rule to get 'b' into the terminal map
    ];
    let parser_with_b = generate_glr_parser(&productions_with_b, 0, None);
    let b_token = *parser_with_b.terminal_map.get_by_left(&regex_name("b")).unwrap();
    let a_token_b = *parser_with_b.terminal_map.get_by_left(&regex_name("a")).unwrap();

    let tokens4 = vec![b_token];
    let mut state4 = parser_with_b.init_glr_parser(None);
    state4.parse(&tokens4);
    // After parsing 'b', there should be no valid active states.
    assert!(!state4.is_ok(), "Parse should fail after invalid token 'b'");

    // Test case 5: "ab"
    let tokens5 = vec![a_token_b, b_token];
    let mut state5 = parser_with_b.init_glr_parser(None);
    state5.parse(&tokens5);
    assert!(!state5.is_ok(), "Parse should fail for 'ab'");
}

fn create_expression_parser() -> GLRParser {
    // S -> E $
    // E -> E + T
    // E -> T
    // T -> T * F
    // T -> F
    // F -> ( E )
    // F -> i
    // This grammar is left-recursive (E->E+T, T->T*F) and has unit productions (E->T, T->F),
    // but does NOT have length-1 cycles.
    let productions = vec![
        prod("S", vec![nt("E"), t("$")]), // Start rule
        prod("E", vec![nt("E"), t("+"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("T"), t("*"), nt("F")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("("), nt("E"), t(")")]),
        prod("F", vec![t("i")]),
    ];
    generate_glr_parser(&productions, 0, None)
}

fn tokenize(parser: &GLRParser, input: &str) -> Vec<TerminalID> {
    input
        .chars()
        .filter_map(|c| {
            parser.terminal_map.get_by_left(&regex_name(&c.to_string()))
                .copied()
        })
        .collect()
}

// --- Tests for Full Parser Generation and Parsing ---

#[test]
fn test_super_simple_grammar() {
    // Grammar: S -> "a" "$"
    let productions = vec![
        prod("S", vec![t("a"), t("$")]), // Start rule
    ];

    let parser = generate_glr_parser(&productions, 0, None);
    println!("Parser: {}", parser);
    
    let a_token = *parser.terminal_map.get_by_left(&regex_name("a")).unwrap();
    let eof_token = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();

    // Test case 1: Valid input "a$"
    let mut state_ok = parser.init_glr_parser(None);
    state_ok.step(a_token);
    state_ok.step(eof_token);
    assert!(state_ok.is_ok(), "Parse should succeed for 'a$'");

    // Test case 2: Invalid input "$"
    let mut state_fail = parser.init_glr_parser(None);
    state_fail.step(eof_token); // wrong token
    assert!(!state_fail.is_ok(), "Parse should fail for input '$'");
}

#[test]
fn test_simple_parse_table_generation_and_parse() {
    // This test now implicitly checks that the simple grammar passes validation.
    let parser = create_simple_parser();
    println!("Parser: {}", parser);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    // dbg!(&parser); // Keep commented unless debugging needed

    let test_cases = [
        ("b", true),
        ("ba", true),
        ("baa", true),
        ("a", false), // Cannot start with 'a'
        ("bb", false), // Cannot have two 'b's
    ];

    for (input, expected_match) in test_cases {
        println!("Testing input: '{}'", input);
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        state.step(eof); // Use step for the final EOF token
        assert_eq!(
            state.is_ok(),
            expected_match,
            "Parse check failed for input: '{}'",
            input
        );
    }
}

#[test]
fn test_expression_parse_table_generation_and_parse() {
    // This test now implicitly checks that the expression grammar passes validation.
    let parser = create_expression_parser();
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    // dbg!(&parser); // Keep commented unless debugging needed

    let test_cases = [
        ("i", true),
        ("i+i*i", true),
        ("i+i", true),
        ("i*i", true),
        ("(i+i)*i", true),
        ("i+", false),   // Incomplete expression
        ("i++i", false), // Invalid syntax
        ("", false),     // Empty input
        (")", false),    // Unmatched parenthesis
        ("(i", false),   // Unmatched parenthesis
    ];

    for (input, expected_match) in test_cases {
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        state.step(eof); // Use step for the final EOF token
        assert_eq!(
            state.is_ok(),
            expected_match,
            "Parse check failed for input: '{}'",
            input
        );
    }
}

#[test]
fn test_unit_production_elimination() {
    // Grammar with unit productions: E -> T, T -> F
    let productions = vec![
        prod("S", vec![nt("E"), t("$")]),
        prod("E", vec![nt("E"), t("+"), nt("T")]),
        prod("E", vec![nt("T")]), // Unit
        prod("T", vec![nt("T"), t("*"), nt("F")]),
        prod("T", vec![nt("F")]), // Unit
        prod("F", vec![t("("), nt("E"), t(")")]),
        prod("F", vec![t("id")]),
    ];

    // --- Generate WITHOUT optimization (by manually calling stages) ---
    let terminal_map = crate::glr::table::assign_terminal_ids(&productions);
    let non_terminal_map = crate::glr::table::assign_non_terminal_ids(&productions);
    let stage_1 = crate::glr::table::stage_1(&productions, 0);
    let stage_2 = crate::glr::table::stage_2(stage_1, &productions);
    let stage_3 = crate::glr::table::stage_3(stage_2, &productions);
    let stage_4 = crate::glr::table::stage_4(stage_3, &productions);
    let stage_5 = crate::glr::table::stage_5(stage_4, &productions, &terminal_map);
    let stage_6 = crate::glr::table::stage_6(stage_5);
    let (unoptimized_table, item_set_map, start_state_id) = crate::glr::table::stage_7(stage_6, &productions, 0, &terminal_map, &non_terminal_map);

    let unoptimized_state_count = unoptimized_table.len();
    let unoptimized_parser = GLRParser { stage_7_table: unoptimized_table, productions: productions.clone(), start_production_id: 0, terminal_map: terminal_map.clone(), non_terminal_map: non_terminal_map.clone(), item_set_map, start_state_id, ignore_terminal_id: None };
    println!("Unoptimized table:");
    println!("{}", unoptimized_parser);

    // --- Generate WITH optimization (using the main function) ---
    let parser = generate_glr_parser(&productions, 0, None);
    let optimized_state_count = parser.stage_7_table.len();
    println!("Optimized table:");
    println!("{}", parser);

    println!("Unoptimized state count: {}", unoptimized_state_count);
    println!("Optimized state count: {}", optimized_state_count);

    // Assert that the optimization reduced the number of states.
    assert!(optimized_state_count < unoptimized_state_count, "Unit production elimination should reduce the number of states.");

    // Assert that the optimized parser still works correctly.
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let tokens = tokenize(&parser, "id+id*id");
    let mut state = parser.init_glr_parser(None);
    state.parse(&tokens);
    state.step(eof);
    assert!(state.is_ok(), "Optimized parser failed to parse a valid expression.");
}

#[test]
fn validation_passes_standard_grammars() {
    // Simple Grammar (already tested implicitly above, but good to be explicit)
    let simple_productions = vec![
        prod("S", vec![nt("A"), t("$")]),
        prod("A", vec![nt("A"), t("a")]),
        prod("A", vec![t("b")]),
    ];
    assert!(analyze::validate(&simple_productions).is_ok());

    // Expression Grammar (already tested implicitly above)
    let expr_productions = vec![
        prod("S", vec![nt("E"), t("$")]),
        prod("E", vec![nt("E"), t("+"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("T"), t("*"), nt("F")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("("), nt("E"), t(")")]),
        prod("F", vec![t("i")]),
    ];
    assert!(analyze::validate(&expr_productions).is_ok());
}

#[test]
fn validation_fails_direct_length_1_recursion() {
    // A -> A
    let productions = vec![
        prod("S", vec![nt("A")]), // Start rule
        prod("A", vec![nt("A")]), // Direct length-1 cycle
        prod("A", vec![t("x")]),  // To make A reachable/productive (though validation runs first)
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Direct length-1 recursion cycle detected: A -> A"));
}

#[test]
fn validation_fails_indirect_length_1_recursion() {
    // A -> B
    // B -> A
    let productions = vec![
        prod("S", vec![nt("A")]), // Start rule
        prod("A", vec![nt("B")]), // A derives B
        prod("B", vec![nt("A")]), // B derives A (cycle)
        prod("A", vec![t("x")]),  // Make productive
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    // The exact path might depend on BTreeSet iteration order, check for involvement
    // With canonicalization, the path should be deterministic.
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("Indirect length-1 recursion cycle detected"));
    assert!(err_msg.contains("A -> B -> A"));

    // A -> B, B -> C, C -> A
    let productions_3 = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("B")]),
        prod("B", vec![nt("C")]),
        prod("C", vec![nt("A")]),
        prod("A", vec![t("x")]),
    ];
    let result_3 = analyze::validate(&productions_3);
     assert!(result_3.is_err());
    let err_msg_3 = result_3.unwrap_err();
    assert!(err_msg_3.contains("Indirect length-1 recursion cycle detected"));
    assert!(err_msg_3.contains("A -> B -> C -> A")); // Order should be deterministic now
}


#[test]
fn validation_fails_direct_length_1_recursion_nullable_prefix() {
    // A -> N A
    // N -> epsilon
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("N"), nt("A")]), // A -> Nullable A (cycle)
        prod("N", vec![]),                // N is nullable
        prod("A", vec![t("x")]),
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Direct length-1 recursion cycle detected: A -> A"));
}

#[test]
fn validation_fails_indirect_length_1_recursion_nullable_prefix() {
    // A -> N B
    // B -> M A
    // N -> epsilon
    // M -> epsilon
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("N"), nt("B")]), // A -> Nullable B
        prod("B", vec![nt("M"), nt("A")]), // B -> Nullable A (cycle)
        prod("N", vec![]),                // N is nullable
        prod("M", vec![]),                // M is nullable
        prod("A", vec![t("x")]),
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("Indirect length-1 recursion cycle detected"));
    assert!(err_msg.contains("A -> B -> A"));
}

#[test]
fn validation_passes_non_unit_recursion() {
    // A -> A t (Not length-1)
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("A"), t("t")]),
        prod("A", vec![t("x")]),
    ];
    assert!(analyze::validate(&productions).is_ok());

    // A -> B A, B -> t (B is not nullable)
    let productions_2 = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("B"), nt("A")]),
        prod("B", vec![t("b")]),
        prod("A", vec![t("x")]),
    ];
     assert!(analyze::validate(&productions_2).is_ok());
}

#[test]
fn validation_fails_left_nullable_left_recursion() {
    // S -> B S a
    // B -> epsilon
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start rule
        prod("S", vec![nt("B"), nt("S"), t("a")]), // Problematic rule
        prod("S", vec![t("b")]),
        prod("B", vec![]), // B is nullable
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("Left-nullable left recursion detected"));
}

#[test]
fn validation_fails_missing_nonterminal() {
    // S -> A
    // A -> B (B is never defined)
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("B")]),
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Non-terminal(s) used in rule RHS but never defined in LHS"));
}

#[test]
fn validation_passes_complex_unit_rules_no_cycle() {
    // S -> A | X
    // A -> B | C
    // B -> D
    // C -> D
    // D -> t
    // X -> Y
    // Y -> t
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("S", vec![nt("X")]),
        prod("A", vec![nt("B")]),
        prod("A", vec![nt("C")]),
        prod("B", vec![nt("D")]),
        prod("C", vec![nt("D")]),
        prod("D", vec![t("t")]),
        prod("X", vec![nt("Y")]),
        prod("Y", vec![t("t")]),
    ];
     assert!(analyze::validate(&productions).is_ok());
}

// --- Tests for remove_productions_with_undefined_nonterminals ---

#[test]
fn test_remove_undefined_simple() {
    // S -> A
    // A -> B (B is undefined)
    // C -> c
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("B")]), // This should be removed
        prod("C", vec![t("c")]),  // This should remain
    ];
    let expected = vec![
        prod("C", vec![t("c")]),
    ];
    let result = remove_productions_with_undefined_nonterminals(&productions, &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_remove_undefined_iterative() {
    // S -> A
    // A -> B
    // B -> C (C is undefined)
    // D -> d
    let productions = vec![
        prod("S", vec![nt("A")]), // Removed in iteration 2 (because A becomes undefined)
        prod("A", vec![nt("B")]), // Removed in iteration 2 (because B becomes undefined)
        prod("B", vec![nt("C")]), // Removed in iteration 1 (because C is undefined)
        prod("D", vec![t("d")]),  // Remains
    ];
    let expected = vec![
        prod("D", vec![t("d")]),
    ];
    let result = remove_productions_with_undefined_nonterminals(&productions, &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_remove_undefined_no_change() {
    // S -> A
    // A -> a
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![t("a")]),
    ];
    let expected = productions.clone();
    let result = remove_productions_with_undefined_nonterminals(&productions, &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_remove_undefined_empty_input() {
    assert!(remove_productions_with_undefined_nonterminals(&[], &[]).is_empty());
}
// --- Tests Demonstrating GLR Capabilities / Limitations ---

#[test]
fn test_ambiguous_dangling_else() {
    // Grammar: Stmt -> if Expr then Stmt
    //                | if Expr then Stmt else Stmt
    //                | other
    //          Expr -> id
    // Input: if id then if id then other else other // This is fine, it's a comment
    // This is ambiguous: the 'else' can attach to the inner or outer 'if'.
    // GLR should *accept* this input by exploring both possibilities.
    let productions = vec![
        prod("S'", vec![nt("Stmt"), t("$")]), // Start
        prod("Stmt", vec![t("if"), nt("Expr"), t("then"), nt("Stmt")]),
        prod("Stmt", vec![t("if"), nt("Expr"), t("then"), nt("Stmt"), t("else"), nt("Stmt")]),
        prod("Stmt", vec![t("other")]), // This is fine, it's a comment
        prod("Expr", vec![t("id")]),
    ];
    let parser = generate_glr_parser(&productions, 0, None);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let tokens = vec![
        *parser.terminal_map.get_by_left(&regex_name("if")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("id")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("then")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("if")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("id")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("then")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("other")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("else")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("other")).unwrap(),
    ];

    let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
    state.parse(&tokens);
    state.step(eof);

    // Limitation/Capability: GLR succeeds because it can handle the shift/reduce conflict
    // by splitting the state. We expect is_ok() to be true.
    // Verifying *both* parse trees were found would require inspecting the GSS
    // or using a non-unit T.
    assert!(state.is_ok(), "GLR parser should accept ambiguous dangling else input");
}

#[test]
fn test_ambiguous_arithmetic() {
    // Grammar: E -> E + E | E * E | id
    // Input: id + id * id
    // This is ambiguous: (id + id) * id or id + (id * id)
    // GLR should accept this.
    let productions = vec![
        prod("S'", vec![nt("E"), t("$")]), // Start
        prod("E", vec![nt("E"), t("+"), nt("E")]),
        prod("E", vec![nt("E"), t("*"), nt("E")]),
        prod("E", vec![t("id")]),
    ];
    let parser = generate_glr_parser(&productions, 0, None);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let tokens = vec![
        *parser.terminal_map.get_by_left(&regex_name("id")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("+")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("id")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("*")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("id")).unwrap(),
    ];

    let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
    state.parse(&tokens);
    state.step(eof);

    // Limitation/Capability: GLR succeeds on ambiguous arithmetic.
    assert!(state.is_ok(), "GLR parser should accept ambiguous arithmetic input");

    let mut state2: GLRParserState<'_> = parser.init_glr_parser(None);
    state2.parse(&tokens);
    state2.step(eof);
    pretty_assertions::assert_eq!(state, state2, "GLR parser should produce the same state for the same input");
    assert_eq!(state, state2, "GLR parser should produce the same state for the same input");
}

#[test]
fn test_reduce_reduce_conflict() {
    // Grammar: S -> A | B
    //          A -> x
    //          B -> x
    // Input: x
    // This grammar has a reduce/reduce conflict on 'x'. // This is fine, it's a comment
    // GLR should handle this by performing both reductions.
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start
        prod("S", vec![nt("A")]),
        prod("S", vec![nt("B")]),
        prod("A", vec![t("x")]),
        prod("B", vec![t("x")]),
    ];
    let parser = generate_glr_parser(&productions, 0, None);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let tokens = vec![*parser.terminal_map.get_by_left(&regex_name("x")).unwrap()];

    let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
    state.parse(&tokens);
    state.step(eof);

    // Limitation/Capability: GLR succeeds despite reduce/reduce conflict.
    assert!(state.is_ok(), "GLR parser should accept input with reduce/reduce conflict");
    // We expect multiple active states before the final step, or merged states in the GSS.
}

#[test]
fn test_epsilon_rules_ambiguity() {
    // Grammar: S -> A B
    //          A -> x | epsilon
    //          B -> x | epsilon
    // Input: x
    // This is ambiguous: S -> A B => x B => x epsilon OR S -> A B => epsilon B => epsilon x
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start
        prod("S", vec![nt("A"), nt("B")]),
        prod("A", vec![t("x")]),
        prod("A", vec![]), // Epsilon
        prod("B", vec![t("x")]),
        prod("B", vec![]), // Epsilon
    ];
    let parser = generate_glr_parser(&productions, 0, None);
    println!("Parser: {}", parser);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let tokens = vec![*parser.terminal_map.get_by_left(&regex_name("x")).unwrap()];
    
    // let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
    // state.step(eof);
    // assert!(state.is_ok(), "GLR parser should accept input with epsilon rules");

    let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
    state.parse(&tokens);
    state.step(eof);

    // Limitation/Capability: GLR handles ambiguity caused by epsilon rules.
    assert!(state.is_ok(), "GLR parser should accept ambiguous input involving epsilon rules");
}

#[test]
fn test_highly_ambiguous_potentially_slow() {
    // Grammar: S -> S S | a
    // Input: aaa
    // This grammar is highly ambiguous (Catalan numbers of parses).
    // GLR should accept it, but performance *could* degrade on larger inputs.
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start
        prod("S", vec![nt("S"), nt("S")]),
        prod("S", vec![t("a")]),
    ];
    let parser = generate_glr_parser(&productions, 0, None);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let tokens = vec![
        *parser.terminal_map.get_by_left(&regex_name("a")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("a")).unwrap(),
        *parser.terminal_map.get_by_left(&regex_name("a")).unwrap(),
    ];

    let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
    state.parse(&tokens);
    state.step(eof);

    // Limitation/Capability: GLR handles highly ambiguous grammars.
    // Performance is a potential limitation not tested for correctness here.
    assert!(state.is_ok(), "GLR parser should accept highly ambiguous S -> S S | a grammar");
}

#[test]
fn test_hidden_left_recursion() {
    // Grammar: S' -> S $
    //          S  -> B S a | b
    //          B  -> epsilon
    // This grammar has hidden left recursion because B is nullable.
    // S -> B S a can effectively act like S -> S a.
    // GLR parsers should handle this correctly.
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start
        prod("S", vec![nt("B"), nt("S"), t("a")]),
        prod("S", vec![t("b")]),
        prod("B", vec![]), // Epsilon
    ];

    // Validation should fail due to left-nullable left recursion
    assert!(analyze::validate(&productions).is_err(), "Validation should fail for left-nullable left recursion");
    // This test case is currently redundant because validation fails before parser generation.
    // It is kept here to document the grammar type but will not be run successfully
    // until the validation logic is adjusted or skipped for testing purposes.
    // If validation is ever removed or changed to allow this, uncomment the rest:
    /*
    let parser = generate_glr_parser(&productions, 0, None); // This will fail due to left-nullable left recursion
    println!("Parser: {}", parser);
    let eof = *parser.terminal_map.get_by_left(&terminal("$")).unwrap();

    let test_cases = [
        ("b", true),    // S -> b
        ("ba", true),   // S -> B S a -> e S a -> S a -> b a
        ("baa", true),  // S -> B S a -> e S a -> S a -> (B S a) a -> (e S a) a -> S a a -> b a a
        ("baaa", true),
        ("a", false),   // Cannot start with 'a'
        ("bb", false),  // Cannot have two 'b's
    ];

    for (input, expected_match) in test_cases {
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        state.step(eof);
        assert_eq!(state.is_ok(), expected_match, "Parse check failed for hidden left recursion input: '{}'", input);
    }
    */
}

#[test]
fn test_right_recursive_grammar_parse() {
    // Grammar: S' -> S $
    //          S  -> 'a' S | 'b'
    // This grammar is right-recursive. The `generate_glr_parser` function should
    // transform it into a left-recursive equivalent, and the resulting parser
    // should correctly parse the language (a*b).
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start rule
        prod("S", vec![t("a"), nt("S")]),
        prod("S", vec![t("b")]),
    ];

    // The generation process automatically resolves the right recursion.
    let parser = generate_glr_parser(&productions, 0, None);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();

    let test_cases = [
        ("b", true),
        ("ab", true),
        ("aab", true),
        ("aaab", true),
        ("a", false),   // Must end in 'b'
        ("ba", false),  // Cannot have 'a' after 'b'
        ("bb", false),  // Cannot have two 'b's
    ];

    for (input, expected_match) in test_cases {
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        state.step(eof);
        assert_eq!(
            state.is_ok(),
            expected_match,
            "Parse check failed for right-recursive input: '{}'",
            input
        );
    }
}

#[test]
fn test_hidden_right_recursion() {
    // Grammar: S' -> S $
    //          S  -> a S B | b
    //          B  -> epsilon
    // This grammar has hidden right recursion because B is nullable.
    // S -> a S B can effectively act like S -> a S.
    // GLR parsers should handle this correctly.
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start
        prod("S", vec![t("a"), nt("S"), nt("B")]),
        prod("S", vec![t("b")]),
        prod("B", vec![]), // Epsilon
    ];

    // Validation should pass as it's not length-1 recursion
    assert!(analyze::validate(&productions).is_ok());

    let parser = generate_glr_parser(&productions, 0, None);
    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();

    let test_cases = [
        ("b", true),    // S -> b
        ("ab", true),   // S -> a S B -> a b B -> a b e
        ("aab", true),  // S -> a S B -> a (a S B) B -> a a b B B -> a a b e e
        ("aaab", true),
        ("a", false),   // Needs a 'b'
        ("ba", false),  // Cannot start with 'b' then 'a'
    ];

    for (input, expected_match) in test_cases {
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        state.step(eof);
        assert_eq!(state.is_ok(), expected_match, "Parse check failed for hidden right recursion input: '{}'", input);
    }
}

#[test]
fn test_nullable_nonterminal_before_terminal() {
    // Grammar:
    // S' ::= A $
    // A  ::= B 'c'
    // B  ::= 'd'
    // B  ::=  (* epsilon *)
    let productions = vec![
        prod("S'", vec![nt("A"), t("$")]), // Start rule
        prod("A", vec![nt("B"), t("c")]),
        prod("B", vec![t("d")]),
        prod("B", vec![]), // Epsilon production for B
    ];

    // Validation should pass for this grammar
    assert!(analyze::validate(&productions).is_ok(), "Validation failed for nullable grammar");

    let parser = generate_glr_parser(&productions, 0, None);
    let eof_token_id = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();
    let c_token_id = *parser.terminal_map.get_by_left(&regex_name("c")).unwrap();
    let d_token_id = *parser.terminal_map.get_by_left(&regex_name("d")).unwrap();
    
    println!("Parser: {}", parser);

    // Test case 1: B -> 'd', so A -> 'd' 'c'
    // Input: "dc$"
    let tokens_dc = vec![d_token_id, c_token_id];
    let mut state_dc: GLRParserState<'_> = parser.init_glr_parser(None);
    state_dc.parse(&tokens_dc);
    state_dc.step(eof_token_id);
    assert!(state_dc.is_ok(), "Parse failed for input 'dc$' (expected A -> d c)");

    // Test case 2: B -> epsilon, so A -> 'c'
    // Input: "c$"
    let tokens_c = vec![c_token_id];
    let mut state_c: GLRParserState<'_> = parser.init_glr_parser(None);
    state_c.parse(&tokens_c);
    state_c.step(eof_token_id);
    assert!(state_c.is_ok(), "Parse failed for input 'c$' (expected A -> epsilon c)");

    // Test case 3: Invalid input "d$" (missing 'c')
    let tokens_d_fail = vec![d_token_id];
    let mut state_d_fail: GLRParserState<'_> = parser.init_glr_parser(None);
    state_d_fail.parse(&tokens_d_fail);
    state_d_fail.step(eof_token_id);
    assert!(!state_d_fail.is_ok(), "Parse succeeded for invalid input 'd$'");

    // Test case 4: Invalid input "$" (A cannot be fully empty)
    let tokens_empty_fail = vec![];
    let mut state_empty_fail: GLRParserState<'_> = parser.init_glr_parser(None);
    state_empty_fail.parse(&tokens_empty_fail);
    state_empty_fail.step(eof_token_id);
    assert!(!state_empty_fail.is_ok(), "Parse succeeded for invalid input '$'");
}

#[test]
fn test_filter_productions_selectivity() {
    // Grammar:
    // S -> X
    // X -> A T_int  (P0) // A is not interesting, T_int is.
    // X -> B        (P1) // B is not interesting, does not lead to T_int.
    // A -> a        (P2) // a is not interesting.
    // B -> b        (P3) // b is not interesting.
    // Goal: If interesting is {T_int}, only X -> A T_int should be kept.
    //       X -> B should be filtered out because its RHS (B) does not lead to T_int.
    let productions = vec![
        prod("S", vec![nt("X")]),
        prod("X", vec![nt("A"), t("T_int")]), // P0
        prod("X", vec![nt("B")]),             // P1
        prod("A", vec![t("a")]),              // P2
        prod("B", vec![t("b")]),              // P3
    ];

    let t_int_symbol = Symbol::Terminal(regex_name("T_int"));
    let interesting_symbols: BTreeSet<Symbol> = [t_int_symbol.clone()].iter().cloned().collect();

    let filtered = filter_productions_by_reachability(&productions, &interesting_symbols);

    // Expected productions to be kept:
    // X -> A T_int (because RHS contains T_int, and X is bootstrap LHS)
    // S -> X       (because X can derive T_int, and S is bootstrap LHS via X)
    // We expect X -> B to be filtered out.
    // We expect A -> a to be filtered out.
    // We expect B -> b to be filtered out.

    let expected_kept_productions = vec![
        // Order might vary based on BTreeSet iteration if not careful,
        // but content should be these two.
        // The filter iterates initial_productions, so order should be preserved if input is ordered.
        prod("S", vec![nt("X")]),
        prod("X", vec![nt("A"), t("T_int")]),
    ];
    
    // Convert to BTreeSet for comparison to ignore order issues if any.
    let filtered_set: BTreeSet<_> = filtered.iter().cloned().collect();
    let expected_set: BTreeSet<_> = expected_kept_productions.iter().cloned().collect();

    assert_eq!(filtered_set.len(), 2, "Expected 2 productions to be kept, got {}. Filtered: {:?}", filtered_set.len(), filtered_set);
    assert_eq!(filtered_set, expected_set, "Filtered productions do not match expected. Filtered: {:?}, Expected: {:?}", filtered_set, expected_set);

    // Specifically check that "X -> B" is NOT in the filtered set.
    let prod_x_b = prod("X", vec![nt("B")]);
    assert!(!filtered_set.contains(&prod_x_b), "Production 'X -> B' should have been filtered out.");
}

#[test]
fn test_standard_expression_grammar_parse() {
    // Grammar:
    // S -> E EOF
    // E -> E PLUS T
    // E -> T
    // T -> T TIMES F
    // T -> F
    // F -> LPAREN E RPAREN
    // F -> I
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]), // Start production (index 0)
        prod("E", vec![nt("E"), t("PLUS"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("T"), t("TIMES"), nt("F")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]),
        prod("F", vec![t("I")]),
    ];

    // println!("Grammar before simplification: {}", display_productions(&productions));
    // println!("Simplified grammar: {}", display_productions(&simplify_grammar(&productions, 0).0));

    // Validate the grammar
    assert!(analyze::validate(&productions).is_ok(), "Validation failed for standard expression grammar");

    let parser = generate_glr_parser(&productions, 0, None);
    println!("Parser: {}", parser); // Useful for debugging the generated table

    // Helper to tokenize space-separated terminal names
    fn tokenize_std_expr(parser: &GLRParser, input_str: &str) -> Vec<TerminalID> {
        input_str.split_whitespace() // This is fine, it's a comment
            .filter_map(|s| parser.terminal_map.get_by_left(&regex_name(s)).copied())
            .collect()
    }

    let test_cases = [
        // Valid inputs
        ("I EOF", true),
        ("I PLUS I EOF", true),
        ("I TIMES I EOF", true),
        ("LPAREN I RPAREN EOF", true),
        ("I PLUS I TIMES I EOF", true), // Should handle precedence via ambiguity resolution in GLR
        ("LPAREN I PLUS I RPAREN TIMES I EOF", true),
        ("I PLUS LPAREN I TIMES I RPAREN EOF", true),
        // Invalid inputs
        ("EOF", false), // E cannot be empty
        ("I PLUS EOF", false), // Missing operand after PLUS
        ("PLUS I EOF", false), // Starts with operator
        ("I I EOF", false), // Missing operator
        ("LPAREN I EOF", false), // Unclosed parenthesis
        ("I RPAREN EOF", false), // Unmatched closing parenthesis
        ("I PLUS TIMES I EOF", false), // Operator sequence
    ];

    for (input_str, expected_match) in test_cases {
        let tokens = tokenize_std_expr(&parser, input_str);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        // No separate EOF step needed as "EOF" is part of the token stream
        assert_eq!(
            state.is_ok(),
            expected_match,
            "Parse check failed for input: '{}'",
            input_str
        );
    }
}

#[test]
fn test_resolve_right_recursion() {
    // Helper to compare production sets regardless of order
    let compare_prods = |p1: &[Production], p2: &[Production]| -> bool {
        let set1: BTreeSet<_> = p1.iter().cloned().collect();
        let set2: BTreeSet<_> = p2.iter().cloned().collect();
        set1 == set2
    };

    // --- Test Case 1: Simple Direct Right-Recursion ---
    let mut prods1 = vec![
        prod("S", vec![t("a"), nt("S")]),
        prod("S", vec![t("b")]),
    ];
    let expected1 = vec![
        prod("S", vec![nt("S_prime"), t("b")]),
        prod("S_prime", vec![nt("S_prime"), t("a")]),
        prod("S_prime", vec![]),
    ];
    analyze::resolve_direct_right_recursion(&mut prods1, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods1, &expected1), "Test 1 failed. Got: {:?}", prods1);

    // --- Test Case 2: No Right-Recursion (should not change) ---
    let mut prods2 = vec![
        prod("S", vec![nt("S"), t("a")]),
        prod("S", vec![t("b")]),
    ];
    let expected2 = prods2.clone();
    analyze::resolve_direct_right_recursion(&mut prods2, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods2, &expected2), "Test 2 failed. Got: {:?}", prods2);

    // --- Test Case 3: Indirect Right-Recursion ---
    let mut prods3 = vec![
        prod("A", vec![t("a"), nt("B")]),
        prod("A", vec![t("c")]),
        prod("B", vec![t("b"), nt("A")]),
        prod("B", vec![t("d")]),
    ];
    // Expected transformation:
    // A is fine.
    // B -> b A | d  (A is substituted)
    // B -> b (a B) | b c | d
    // B -> b a B | b c | d (Now B has direct right-recursion)
    // B -> B' (b c) | B' d
    // B' -> B' (b a) | ε
    let expected3 = vec![
        prod("A", vec![t("a"), nt("B")]),
        prod("A", vec![t("c")]),
        prod("B", vec![nt("B_prime"), t("b"), t("c")]),
        prod("B", vec![nt("B_prime"), t("d")]),
        prod("B_prime", vec![nt("B_prime"), t("b"), t("a")]),
        prod("B_prime", vec![]),
    ];
    analyze::resolve_direct_right_recursion(&mut prods3, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods3, &expected3), "Test 3 failed. Got: {:?}", prods3);

    // --- Test Case 4: Hidden Direct Right-Recursion ---
    let mut prods4 = vec![
        prod("S", vec![t("a"), nt("S"), nt("N")]),
        prod("S", vec![t("b")]),
        prod("N", vec![]),
    ];
    let expected4 = vec![
        prod("S", vec![nt("S_prime"), t("b")]),
        prod("S_prime", vec![nt("S_prime"), t("a"), nt("N")]),
        prod("S_prime", vec![]),
        prod("N", vec![]),
    ];
    analyze::resolve_direct_right_recursion(&mut prods4, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods4, &expected4), "Test 4 failed. Got: {:?}", prods4);

    // --- Test Case 5: Multiple Non-Recursive Choices ---
    let mut prods5 = vec![
        prod("S", vec![t("a"), nt("S")]),
        prod("S", vec![t("b")]),
        prod("S", vec![t("c")]),
    ];
    let expected5 = vec![
        prod("S", vec![nt("S_prime"), t("b")]),
        prod("S", vec![nt("S_prime"), t("c")]),
        prod("S_prime", vec![nt("S_prime"), t("a")]),
        prod("S_prime", vec![]),
    ];
    analyze::resolve_direct_right_recursion(&mut prods5, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods5, &expected5), "Test 5 failed. Got: {:?}", prods5);

    // --- Test Case 6: Multiple Recursive Choices ---
    let mut prods6 = vec![
        prod("S", vec![t("a"), nt("S")]),
        prod("S", vec![t("b"), nt("S")]),
        prod("S", vec![t("c")]),
    ];
    let expected6 = vec![
        prod("S", vec![nt("S_prime"), t("c")]),
        prod("S_prime", vec![nt("S_prime"), t("a")]),
        prod("S_prime", vec![nt("S_prime"), t("b")]),
        prod("S_prime", vec![]),
    ];
    analyze::resolve_direct_right_recursion(&mut prods6, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods6, &expected6), "Test 6 failed. Got: {:?}", prods6);

    // --- Test Case 7: Both Left and Right Recursive ---
    // S -> S S | a
    // Should eliminate the right-recursion part.
    let mut prods7 = vec![
        prod("S", vec![nt("S"), nt("S")]),
        prod("S", vec![t("a")]),
    ];
    let expected7 = vec![
        prod("S", vec![nt("S_prime"), t("a")]),
        prod("S_prime", vec![nt("S_prime"), nt("S")]),
        prod("S_prime", vec![]),
    ];
    analyze::resolve_direct_right_recursion(&mut prods7, |name| format!("{}_prime", name));
    assert!(compare_prods(&prods7, &expected7), "Test 7 failed. Got: {:?}", prods7);
}

#[test]
fn test_explain_stack() {
    let parser = create_simple_parser();
    // Stack for input "b" would be something like:
    // 1. Start in State 0.
    // 2. Shift 'b', go to a new state. (Stack: [0, X])
    
    // Let's find the actual state IDs from the generated parser.
    let start_state = parser.start_state_id;
    let b_token_id = *parser.terminal_map.get_by_left(&regex_name("b")).unwrap();
    
    let start_row = &parser.stage_7_table[&start_state];
    let shift_action = &start_row.shifts_and_reduces_full[&b_token_id];

    let state_after_b = match shift_action {
        crate::glr::table::Stage7ShiftsAndReducesLookaheadValue::Shift(id) => *id,
        _ => panic!("Expected shift on 'b' from start state"),
    };

    let stack_to_explain = vec![start_state, state_after_b];
    let explanation = parser.explain_stack(&stack_to_explain);

    println!("{}", explanation);

    // Assertions for State 0 (start state)
    assert!(explanation.contains(&format!("State {}:", start_state.0)));
    assert!(explanation.contains("Items:"));
    assert!(explanation.contains("[S -> • A '$', $]"));
    assert!(explanation.contains("On 'b': Shift to State"));

    // Assertions for State after shifting 'b'
    assert!(explanation.contains(&format!("State {}:", state_after_b.0)));
    assert!(explanation.contains("[A -> 'b' •, $]"));
    assert!(explanation.contains("On '$': Reduce by rule"));
}

#[test]
#[ignore = "Direct left recursion resolution avoids this shift/reduce conflict."]
fn test_parser_stats_conflicts() {
    // --- Test Reduce/Reduce Conflict ---
    // Grammar: S -> A | B, A -> x, B -> x
    // This has a reduce/reduce conflict on 'x'.
    let rr_productions = vec![
        prod("S'", vec![nt("S"), t("$")]),
        prod("S", vec![nt("A")]),
        prod("S", vec![nt("B")]),
        prod("A", vec![t("x")]),
        prod("B", vec![t("x")]),
    ];
    let rr_parser = generate_glr_parser(&rr_productions, 0, None);
    let rr_stats = stats::get_stats(&rr_parser);

    println!("Reduce/Reduce Conflict Parser Stats:\n{}", rr_stats);
    assert_eq!(rr_stats.num_shift_reduce_conflicts, 0);
    assert_eq!(rr_stats.num_reduce_reduce_conflicts, 1, "Expected one R/R conflict");

    // --- Test Shift/Reduce Conflict ---
    // Grammar: Stmt -> if E then S | if E then S else S | other
    // This has a shift/reduce conflict on 'else'.
    let sr_productions = vec![
        prod("S'", vec![nt("Stmt"), t("$")]),
        prod("Stmt", vec![t("if"), nt("Expr"), t("then"), nt("Stmt")]),
        prod("Stmt", vec![t("if"), nt("Expr"), t("then"), nt("Stmt"), t("else"), nt("Stmt")]),
        prod("Stmt", vec![t("other")]),
        prod("Expr", vec![t("id")]),
    ];
    let sr_parser = generate_glr_parser(&sr_productions, 0, None);
    let sr_stats = stats::get_stats(&sr_parser);

    println!("Shift/Reduce Conflict Parser Stats:\n{}", sr_stats);
    assert_eq!(sr_stats.num_shift_reduce_conflicts, 1, "Expected one S/R conflict");
    assert_eq!(sr_stats.num_reduce_reduce_conflicts, 0);
}
// --- Notes on Limitations Not Easily Tested Here ---

#[test]
fn test_lr1_not_lalr1_grammar() {
    // This grammar is a classic example of a grammar that is LR(1) but not LALR(1).
    // An LALR(1) parser would merge states and create a reduce/reduce conflict.
    // A canonical LR(1) parser should handle it without conflicts.
    // S' -> S $
    // S  -> a E c | a F d | b F c | b E d
    // E  -> e
    // F  -> e
    let productions = vec![
        prod("S'", vec![nt("S"), t("$")]), // Start rule
        prod("S", vec![t("a"), nt("E"), t("c")]),
        prod("S", vec![t("a"), nt("F"), t("d")]),
        prod("S", vec![t("b"), nt("F"), t("c")]),
        prod("S", vec![t("b"), nt("E"), t("d")]),
        prod("E", vec![t("e")]),
        prod("F", vec![t("e")]),
    ];

    // Validation should pass
    assert!(analyze::validate(&productions).is_ok());

    // Parser generation should succeed without conflicts for an LR(1) generator
    let parser = generate_glr_parser(&productions, 0, None);
    println!("Parser: {}", parser); // Useful for debugging the generated table

    // Check stats to be sure there are no conflicts
    let stats = stats::get_stats(&parser);
    assert_eq!(stats.num_reduce_reduce_conflicts, 0, "LR(1) parser should not have R/R conflicts for this grammar");
    assert_eq!(stats.num_shift_reduce_conflicts, 0, "LR(1) parser should not have S/R conflicts for this grammar");

    let eof = *parser.terminal_map.get_by_left(&regex_name("$")).unwrap();

    let test_cases = [("aec", true), ("afd", false), ("bfc", false), ("bed", true), ("aed", true), ("afc", false), ("bec", true), ("bfd", false), ("e", false), ("ac", false)];

    for (input, expected_match) in test_cases {
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_> = parser.init_glr_parser(None);
        state.parse(&tokens);
        state.step(eof);
        assert_eq!(state.is_ok(), expected_match, "Parse check failed for LR(1)-specific input: '{}'", input);
    }
}

// 1. Semantic Ambiguity: These tests use T=(), so while the parser finds *a* parse (or confirms
//    parsability) for ambiguous grammars, they don't demonstrate *how* multiple semantic
//    results (parse trees) would be represented or combined. A more complex `MergeAndIntersect`
//    implementation for T would be needed to show this.
// 2. Performance: While `test_highly_ambiguous_potentially_slow` uses a grammar known for
//    exponential ambiguity, verifying performance limits requires benchmarking, not just correctness checks.
// 3. Error Reporting: The current tests check `is_ok()`. A limitation could be the quality/detail
//    of error reporting when `is_ok()` is false (e.g., pinpointing the error location).
// 4. Validation Scope: The `analyze::validate` function currently checks for missing non-terminals
//    and length-1 cycles. It doesn't detect all potential issues like useless rules (unreachable
//    or non-productive non-terminals), which could be considered a limitation of the validation step.
