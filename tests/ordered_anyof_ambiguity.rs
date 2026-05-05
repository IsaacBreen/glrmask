use glrmask::{Constraint, ConstraintState, Vocab};
use serde_json::{json, Map, Value};

fn make_byte_vocab() -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = (0u32..=255).map(|byte| (byte, vec![byte as u8])).collect();
    Vocab::new(entries, None)
}

fn ordered_name(position: usize, stem: &str) -> String {
    format!("{position:02}_{stem}")
}

struct FamilySpec {
    order: Vec<String>,
    common_required: Vec<String>,
    capability_keys: Vec<String>,
}

fn one_hole_then_suffix_family(n_caps: usize) -> FamilySpec {
    let common_required = vec![
        ordered_name(0, "req0"),
        ordered_name(1, "req1"),
        ordered_name(3, "req2"),
        ordered_name(4, "req3"),
        ordered_name(5, "req4"),
    ];
    let capability_keys = (0..n_caps)
        .map(|index| {
            let position = if index == 0 { 2 } else { 5 + index };
            ordered_name(position, &format!("cap{index:02}"))
        })
        .collect::<Vec<_>>();

    let mut order = common_required.clone();
    order.extend(capability_keys.iter().cloned());
    order.sort();

    FamilySpec {
        order,
        common_required,
        capability_keys,
    }
}

fn tiny_array_schema() -> Value {
    json!({
        "type": "array",
        "items": { "enum": ["x"] },
        "minItems": 1,
        "maxItems": 1,
    })
}

fn tiny_array_text() -> &'static str {
    r#"["x"]"#
}

fn singleton_required_anyof_schema(spec: &FamilySpec) -> String {
    let properties = spec
        .order
        .iter()
        .map(|key| (key.clone(), tiny_array_schema()))
        .collect::<Map<String, Value>>();
    let required = spec
        .common_required
        .iter()
        .cloned()
        .map(Value::String)
        .collect::<Vec<_>>();
    let any_of = spec
        .capability_keys
        .iter()
        .map(|key| json!({ "required": [key] }))
        .collect::<Vec<_>>();

    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
        "required": required,
        "anyOf": any_of,
    })
    .to_string()
}

fn example_from_capability_mask(spec: &FamilySpec, capability_mask: usize) -> String {
    let mut present = spec.common_required.clone();
    for (index, key) in spec.capability_keys.iter().enumerate() {
        if ((capability_mask >> index) & 1) != 0 {
            present.push(key.clone());
        }
    }

    let fields = spec
        .order
        .iter()
        .filter(|key| present.iter().any(|present_key| present_key == *key))
        .map(|key| format!("\"{key}\": {}", tiny_array_text()))
        .collect::<Vec<_>>();

    format!("{{{}}}", fields.join(", "))
}

fn stack_count(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

fn measure_max_stack_and_path_counts(constraint: &Constraint, text: &str) -> (usize, usize) {
    let mut state = constraint.start();
    let mut max_stacks = stack_count(&state);
    let mut max_paths = state.parser_path_count(1_000_000);

    for (index, &byte) in text.as_bytes().iter().enumerate() {
        state.commit_bytes(&[byte]).unwrap_or_else(|err| {
            panic!(
                "example replay failed at byte_index={index} byte={byte:?} char={} for text={text}: {err}",
                byte as char
            )
        });
        max_stacks = max_stacks.max(stack_count(&state));
        max_paths = max_paths.max(state.parser_path_count(1_000_000));
    }

    (max_stacks, max_paths)
}

// Minimal reproducer for the ordered-object + singleton-required-anyOf ambiguity
// behind the o21108 family. The growth comes from shared ordered properties plus
// singleton anyOf requirements, not from array min/max cardinality itself.
#[test]
fn ordered_anyof_singleton_required_ambiguity_grows() {
    let vocab = make_byte_vocab();
    let cases = [
        (2usize, 0b01usize, 2usize),
        (5usize, 0b01111usize, 3usize),
        (8usize, 0b00110011usize, 5usize),
        (10usize, 0b0000110101usize, 6usize),
    ];

    for (n_caps, capability_mask, expected_max) in cases {
        let spec = one_hole_then_suffix_family(n_caps);
        let schema = singleton_required_anyof_schema(&spec);
        let example = example_from_capability_mask(&spec, capability_mask);
        let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
        let (max_stacks, max_paths) = measure_max_stack_and_path_counts(&constraint, &example);

        println!(
            "n_caps={n_caps} mask={capability_mask:0width$b} max_stacks={max_stacks} max_paths={max_paths}",
            width = n_caps,
        );

        assert_eq!(
            max_stacks, expected_max,
            "unexpected max stack count for n_caps={n_caps} mask={capability_mask:0width$b}",
            width = n_caps,
        );
        assert_eq!(
            max_paths, expected_max,
            "unexpected max path count for n_caps={n_caps} mask={capability_mask:0width$b}",
            width = n_caps,
        );
        assert_eq!(
            max_stacks, max_paths,
            "max stack and path counts should match for n_caps={n_caps} mask={capability_mask:0width$b}",
            width = n_caps,
        );
    }
}