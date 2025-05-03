use crate::glr::grammar::{nt, prod, t, NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::{GLRParser, GLRParserState};
use crate::glr::table::{generate_glr_parser, TerminalID};
use crate::glr::analyze; // Import the analyze module
use bimap::BiBTreeMap;

// --- Helper Functions for Tests ---

fn create_simple_parser() -> GLRParser {
    // S -> A $
    // A -> A a
    // A -> b
    // This grammar is left-recursive but does NOT have length-1 cycles.
    let productions = vec![
        prod("S", vec![nt("A"), t("$")]), // Start rule
        prod("A", vec![nt("A"), t("a")]),
        prod("A", vec![t("b")]),
    ];

    generate_glr_parser(&productions, 0)
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
    generate_glr_parser(&productions, 0)
}

fn tokenize(parser: &GLRParser, input: &str) -> Vec<TerminalID> {
    input
        .chars()
        .filter_map(|c| {
            parser
                .terminal_map
                .get_by_left(&Terminal(c.to_string()))
                .copied()
        })
        .collect()
}

// --- Tests for Full Parser Generation and Parsing ---

#[test]
fn test_simple_parse_table_generation_and_parse() {
    // This test now implicitly checks that the simple grammar passes validation.
    let parser = create_simple_parser();
    let eof = *parser
        .terminal_map
        .get_by_left(&Terminal("$".to_string()))
        .unwrap();
    // dbg!(&parser); // Keep commented unless debugging needed

    let test_cases = [
        ("b", true),
        ("ba", true),
        ("baa", true),
        ("a", false), // Cannot start with 'a'
        ("bb", false), // Cannot have two 'b's
    ];

    for (input, expected_match) in test_cases {
        let tokens = tokenize(&parser, input);
        let mut state: GLRParserState<'_, ()> = parser.init_glr_parser();
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
    let eof = *parser
        .terminal_map
        .get_by_left(&Terminal("$".to_string()))
        .unwrap();
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
        let mut state: GLRParserState<'_, ()> = parser.init_glr_parser();
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

// --- Tests Specifically for Grammar Validation Logic ---

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
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("Indirect length-1 recursion cycle detected"));
    assert!(err_msg.contains("A -> B -> A") || err_msg.contains("B -> A -> B"));

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
    // Depending on traversal order, the cycle could start from A or B
    assert!(err_msg.contains("A -> B -> A") || err_msg.contains("B -> A -> B"));
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
fn validation_fails_missing_nonterminal() {
    // S -> A
    // A -> B (B is never defined)
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![nt("B")]),
    ];
    let result = analyze::validate(&productions);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Non-terminal(s) used in rule RHS but never defined in LHS: {\"B\"}"));
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