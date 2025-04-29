use crate::glr::grammar::{nt, prod, t, Terminal};
use crate::glr::parser::{GLRParser, GLRParserState};
use crate::glr::table::{generate_glr_parser, TerminalID};
use bimap::BiBTreeMap;

fn create_simple_parser() -> GLRParser {
    let productions = vec![
        prod("S", vec![nt("A"), t("$")]),
        prod("A", vec![nt("A"), t("a")]),
        prod("A", vec![t("b")]),
    ];

    generate_glr_parser(&productions, 0)
}

fn create_expression_parser() -> GLRParser {
    let productions = vec![
        prod("S", vec![nt("E"), t("$")]),
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
    input.chars()
        .filter_map(|c| parser.terminal_map.get_by_left(&Terminal(c.to_string())).copied())
        .collect()
}

#[test]
fn test_simple_parse_table() {
    let parser = create_simple_parser();
    let eof = *parser.terminal_map.get_by_left(&Terminal("$".to_string())).unwrap();
    dbg!(&parser);

    let test_cases = [
        ("b", true),
        ("ba", true),
        ("baa", true),
        ("a", false),
        ("bb", false),
    ];

    for (input, expected_match) in test_cases {
        let mut state: GLRParserState<'_, ()> = parser.init_glr_parser();
        state.parse(&tokenize(&parser, input));
        state.parse(&[eof]);
        assert_eq!(state.is_ok(), expected_match, "Failed for input: {}", input);
    }
}

#[test]
fn test_parse_simple_expression() {
    let parser = create_expression_parser();
    let eof = *parser.terminal_map.get_by_left(&Terminal("$".to_string())).unwrap();
    dbg!(&parser);

    let test_cases = [
        ("i", true),
        ("i+i*i", true),
        ("i+i", true),
        ("i*i", true),
        ("(i+i)*i", true),
        ("i+", false),
        ("i++i", false),
        ("", false),
        (")", false),
    ];

    for (input, expected_match) in test_cases {
        let mut state: GLRParserState<'_, ()> = parser.init_glr_parser();
        state.parse(&tokenize(&parser, input));
        state.parse(&[eof]);
        assert_eq!(
            state.is_ok(),
            expected_match, 
            "Failed for input: {}", input
        );
    }
}