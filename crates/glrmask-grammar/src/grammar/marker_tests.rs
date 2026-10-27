//! Source markers are optional, non-lexical compiler probes. These tests do not
//! rely on LR construction or runtime observers.
use super::{ast, flat::Terminal, glrm::ParsedComponent};

fn lower(source: &str) -> ast::NamedGrammar {
    ParsedComponent::parse_glrm(source).unwrap().lower_closed().unwrap()
}

#[test]
fn marker_source_roundtrips_without_becoming_a_token_or_double_optional() {
    let named = lower("start root\nnt root = @mark(7) \"a\" @mark(4294967295)\n");
    let dump = super::glrm::to_glrm_v1(&named).unwrap();
    assert!(dump.contains("@mark(7)"));
    assert!(!dump.contains("@mark(7)?"));
    assert_eq!(dump, super::glrm::to_glrm_v1(&lower(&dump)).unwrap());
    let flat = ast::lower(&named).unwrap();
    let ids = flat.terminals.iter().filter_map(|t| match t {
        Terminal::MarkerProbe { marker_id, .. } => Some(*marker_id), _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(ids, [7, u32::MAX]);
    assert!(!flat.start_is_nullable());
    assert!(flat.terminals.iter().all(|t| !matches!(t,
        Terminal::SpecialToken { .. } | Terminal::SpecialTokenRange { .. })));
}

#[test]
fn marker_only_rule_is_nullable_but_its_query_terminal_is_not() {
    let flat = ast::lower(&lower("start root\nnt root = @mark(0)\n")).unwrap();
    assert!(flat.start_is_nullable());
    assert_eq!(flat.terminals.len(), 1);
    assert!(matches!(flat.terminals[0], Terminal::MarkerProbe { marker_id: 0, .. }));
    assert!(!flat.terminals[0].is_nullable());
}

#[test]
fn repeated_marker_id_shares_one_probe() {
    let flat = ast::lower(&lower("start root\nnt root = (@mark(42) \"x\")* @mark(42)\n")).unwrap();
    assert!(flat.start_is_nullable());
    assert_eq!(flat.terminals.iter().filter(|t| matches!(t,
        Terminal::MarkerProbe { marker_id: 42, .. })).count(), 1);
}

#[test]
fn markers_survive_inline_component_lowering() {
    let flat = ast::lower(&lower(
        "start root\nnt root = @mark(1) child @mark(2)\ngrammar child {\nstart item\nnt item = @mark(3) \"x\"\n}\n"
    )).unwrap();
    let mut ids = flat.terminals.iter().filter_map(|t| match t {
        Terminal::MarkerProbe { marker_id, .. } => Some(*marker_id), _ => None,
    }).collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, [1, 2, 3]);
}

#[test]
fn markers_reject_invalid_ids_and_byte_terminal_contexts() {
    for atom in ["@mark(-1)", "@mark(4294967296)", "@mark(name)", "@mark()", "@mark(1,2)"] {
        assert!(ParsedComponent::parse_glrm(&format!("start root\nnt root = {atom}\n")).is_err(), "{atom}");
    }
    let result = ParsedComponent::parse_glrm("start root\nt ITEM = @mark(1)\nnt root = ITEM\n")
        .and_then(|p| p.lower_closed()).and_then(|n| ast::lower(&n));
    assert!(result.is_err(), "a marker must not be treated as an empty byte regex");
    assert!(ParsedComponent::parse_glrm("start root\nnt root = @token(1)\n").is_err());
}

#[test]
fn ebnf_and_lark_support_the_same_marker_atom() {
    let ebnf = crate::parse_ebnf_to_named("root ::= @mark(7) \"a\" @mark(8)").unwrap();
    let flat = ast::lower(&ebnf).unwrap();
    assert_eq!(flat.terminals.iter().filter(|t| matches!(t, Terminal::MarkerProbe { .. })).count(), 2);
    let lark = "start: @mark(7) \"a\" @mark(8)";
    let err = crate::parse_lark_document(lark).unwrap_err();
    assert!(err.to_string().contains("`@mark` isn't supported"));
}
