use glrmask::{Constraint, ConstraintState, Vocab};
use glrmask::__private::ConstraintStateExt as _;

// The second root key is load-bearing. It is globally subtracted from the shared
// additional-property key terminal, then added back inside "FAQs", where it is
// not a fixed key. At the next-key quote, that addback path is still viable
// alongside the fixed "FAQs" keys.
const O15627_SCHEMA_MRE: &str = r#"
{
  "type": "object",
  "properties": {
    "FAQs": {
      "type": "object",
      "properties": {
        "a1": {"type": "string"},
        "a10": {"type": "string"},
        "a2": {"type": "string"},
        "a3": {"type": "string"},
        "a4": {"type": "string"},
        "a5": {"type": "string"},
        "a6": {"type": "string"},
        "a7": {"type": "string"},
        "a8": {"type": "string"},
        "a9": {"type": "string"},
        "msg": {"type": "string"},
        "q1": {"type": "string"},
        "q10": {"type": "string"},
        "q2": {"type": "string"},
        "q3": {"type": "string"},
        "q4": {"type": "string"},
        "q5": {"type": "string"},
        "q6": {"type": "string"},
        "q7": {"type": "string"},
        "q8": {"type": "string"},
        "q9": {"type": "string"},
        "title": {"type": "string"}
      }
    },
    "dummy": true
  }
}
"#;

const O15627_PREFIX: &[u8] = br#"{"FAQs": {"a1": "This is the answer to question 1.", ""#;

fn bytes_vocab() -> Vocab {
    Vocab::new((0u8..=255).map(|b| (b as u32, vec![b])).collect(), None)
}

fn stack_count(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

#[test]
fn o15627_schema_prefix_has_single_gss_path_at_next_key_quote() {
    let constraint = Constraint::from_json_schema(O15627_SCHEMA_MRE, &bytes_vocab()).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(O15627_PREFIX).unwrap();

    let stacks = state.debug_parser_stacks();
    assert_eq!(state.parser_path_count(1_000_000), 1, "{stacks:?}");
    assert_eq!(stack_count(&state), 1, "{stacks:?}");
}
