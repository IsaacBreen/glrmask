use glrmask::{Constraint, ConstraintState, Vocab};
use glrmask::__private::ConstraintStateExt as _;
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

fn minimized_stack_growth_schema(n_props: usize) -> String {
    let mut properties = Map::new();
    for index in 1..=n_props {
        properties.insert(format!("a{index}"), json!({ "type": "string" }));
    }
    properties.insert("y".to_string(), json!({ "type": "array" }));

    json!({
        "type": "object",
        "patternProperties": {},
        "properties": properties,
        "anyOf": [
            { "properties": { "x": { "type": "array" } } },
            { "properties": { "y": { "type": "array" } } },
        ],
    })
    .to_string()
}

fn minimized_stack_growth_example(n_props: usize) -> String {
    let mut fields = Vec::with_capacity(n_props + 1);
    for index in 1..=n_props {
        fields.push(format!("\"a{index}\": \"z\""));
    }
    fields.push("\"y\": [\"z\"]".to_string());
    format!("{{{}}}", fields.join(", "))
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

fn max_stack_depth(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .flat_map(|(_, stacks)| stacks.iter().map(|(stack, _)| stack.len()))
        .max()
        .unwrap_or(0)
}

fn stack_depth_before_closing_suffix(constraint: &Constraint, text: &str) -> usize {
    let mut state = constraint.start();
    let prefix = text
        .strip_suffix(r#""]}"#)
        .expect("example should end with closing \"]}\" suffix");

    for (index, &byte) in prefix.as_bytes().iter().enumerate() {
        state.commit_bytes(&[byte]).unwrap_or_else(|err| {
            panic!(
                "prefix replay failed at byte_index={index} byte={byte:?} char={} for text={text}: {err}",
                byte as char
            )
        });
    }

    max_stack_depth(&state)
}

// Minimal reproducer for the ordered-object + singleton-required-anyOf ambiguity
// behind the o21108 family. This should remain bounded even as capability
// branches are added.
#[test]
fn ordered_anyof_singleton_required_ambiguity_stays_bounded() {
    let vocab = make_byte_vocab();
    let cases = [
        (2usize, 0b01usize, 2usize),
        (5usize, 0b01111usize, 2usize),
        (8usize, 0b00110011usize, 2usize),
        (10usize, 0b0000110101usize, 2usize),
    ];

    for (n_caps, capability_mask, max_allowed) in cases {
        let spec = one_hole_then_suffix_family(n_caps);
        let schema = singleton_required_anyof_schema(&spec);
        let example = example_from_capability_mask(&spec, capability_mask);
        let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
        let (max_stacks, max_paths) = measure_max_stack_and_path_counts(&constraint, &example);

        println!(
            "n_caps={n_caps} mask={capability_mask:0width$b} max_stacks={max_stacks} max_paths={max_paths}",
            width = n_caps,
        );

        assert!(
            max_stacks <= max_allowed,
            "unexpected max stack count {max_stacks} > {max_allowed} for n_caps={n_caps} mask={capability_mask:0width$b}",
            width = n_caps,
        );
        assert!(
            max_paths <= max_allowed,
            "unexpected max path count {max_paths} > {max_allowed} for n_caps={n_caps} mask={capability_mask:0width$b}",
            width = n_caps,
        );
        assert_eq!(
            max_stacks, max_paths,
            "max stack and path counts should match for n_caps={n_caps} mask={capability_mask:0width$b}",
            width = n_caps,
        );
    }
}

#[test]
fn minimized_anyof_pattern_object_stack_depth_stays_bounded() {
    let vocab = make_byte_vocab();
    let cases = [5usize, 35usize];
    let mut measured = Vec::new();

    for n_props in cases {
        let schema = minimized_stack_growth_schema(n_props);
        let example = minimized_stack_growth_example(n_props);
        let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
        let depth = stack_depth_before_closing_suffix(&constraint, &example);

        println!("n_props={n_props} depth_before_suffix={depth}");
        assert!(depth > 0, "expected a live parser stack before the closing suffix for n_props={n_props}");
        measured.push((n_props, depth));
    }

    let depth_at_5 = measured[0].1;
    let depth_at_35 = measured[1].1;
    assert!(
        depth_at_35 <= depth_at_5 + 2,
        "stack depth should remain nearly flat across sibling growth; n=5 depth={depth_at_5}, n=35 depth={depth_at_35}"
    );
    assert!(
        depth_at_35 <= 16,
        "stack depth should stay absolutely bounded at the closing suffix; n=35 depth={depth_at_35}"
    );
}

fn build_config_schema_with_optional_environment_branches() -> &'static str {
    r#"{
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "context": {"type": "string"},
                                "image": {"type": "string"},
                                "sync": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"}
                                }
                            }
                        }
                    },
                    "tagPolicy": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "gitCommit": {}
                        }
                    }
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "context": {"type": "string"},
                                "image": {"type": "string"},
                                "sync": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"}
                                }
                            }
                        }
                    },
                    "local": {},
                    "tagPolicy": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "gitCommit": {}
                        }
                    }
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "context": {"type": "string"},
                                "image": {"type": "string"},
                                "sync": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"}
                                }
                            }
                        }
                    },
                    "googleCloudBuild": {},
                    "tagPolicy": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "gitCommit": {}
                        }
                    }
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "context": {"type": "string"},
                                "image": {"type": "string"},
                                "sync": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"}
                                }
                            }
                        }
                    },
                    "cluster": {},
                    "tagPolicy": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "gitCommit": {}
                        }
                    }
                }
            }
        ]
    }"#
}

#[test]
fn build_config_optional_environment_branches_collapse_stack_continuations() {
    let vocab = make_byte_vocab();
    let constraint = Constraint::from_json_schema(
        build_config_schema_with_optional_environment_branches(),
        &vocab,
    )
    .unwrap();
    let example = r#"{"artifacts": [{"context": ".", "image": "gcr.io/k8s-skaffold/example", "sync": {"*.py": ".", "css/**/*.css": "app/css"}}], "tagPolicy": {"gitCommit": {}}}"#;

    let (max_stacks, max_paths) = measure_max_stack_and_path_counts(&constraint, example);
    assert_eq!((max_stacks, max_paths), (1, 1));
}
