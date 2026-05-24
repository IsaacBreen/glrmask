use glrmask::{Constraint, ConstraintState, Vocab};

const GENERIC_FIXED_KEY_SPLIT_GLRM: &str = r#"
start start;

t JSON_STRING ::= /"(?:[^\x00-\x1f\x7f"\\]|\\["\\bfnrt])*"/;
t sep ::= /, /;

nt json_member ::= JSON_STRING ": " json_value;
nt json_object ::= "{" (json_member (sep json_member)*)? "}";
nt json_value ::= JSON_STRING | json_object;

nt fixed_faqs ::= "\"FAQs\": " "{" "\"a1\": " JSON_STRING (sep "\"a10\": " JSON_STRING)?;
nt start ::= "{" (fixed_faqs | JSON_STRING ": " json_value) "}";
"#;

const INPUT: &str = r#"{"FAQs": {"a1": "x.", ""#;

fn byte_vocab() -> Vocab {
    Vocab::new((0u8..=127).map(|byte| (byte as u32, vec![byte])).collect(), None)
}

fn live_stack_count(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

#[test]
fn generic_property_and_fixed_key_paths_survive_to_next_key_prefix() {
    let constraint = Constraint::from_glrm_grammar(GENERIC_FIXED_KEY_SPLIT_GLRM, &byte_vocab())
        .unwrap();
    let mut state = constraint.start();
    let mut trace = Vec::new();

    for (byte_index, &byte) in INPUT.as_bytes().iter().enumerate() {
        state.commit_bytes(&[byte]).unwrap();
        trace.push((
            byte_index,
            byte,
            state.parser_path_count(1_000_000),
            live_stack_count(&state),
        ));
    }

    // The split is between the fixed "FAQs" property path and a generic
    // JSON_STRING property path. It then survives through the nested string
    // value, comma separator, and the quote that starts the next nested key.
    assert_eq!(trace[6], (6, b'"', 2, 2));
    assert_eq!(trace[22], (22, b'"', 2, 2));
}
