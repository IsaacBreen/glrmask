//! General integer divisibility may not silently degrade to a range or wildcard.
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, Vocab};

fn allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32).is_some_and(|word| word & (1 << (id % 32)) != 0)
}

#[test]
fn integer_multiple_exact_public_masks_and_artifact_roundtrip() {
    let values = (-64i64..=64).collect::<Vec<_>>();
    let vocab = Vocab::new(values.iter().enumerate().map(|(id, value)| {
        (id as u32, format!("{{\"v\": {value}}}").into_bytes())
    }).collect::<Vec<_>>());
    for divisor in [3i64, 12, 37] {
        for minimum in [None, Some(0)] {
            let mut number = serde_json::json!({"type": "integer", "multipleOf": divisor});
            if let Some(minimum) = minimum {
                number["minimum"] = minimum.into();
            }
            let schema = serde_json::json!({
                "type": "object", "properties": {"v": number},
                "required": ["v"], "additionalProperties": false,
            }).to_string();
            for mode in [Optimization::Auto, Optimization::FastBuild, Optimization::FastRuntime] {
                let constraint = Grammar::from_json_schema(&schema)
                    .compile_with(&vocab, BuildOptions::default().optimization(mode)).unwrap();
                let loaded = Constraint::load(constraint.save()).unwrap();
                for instance in [&constraint, &loaded] {
                    let mask = instance.start().mask();
                    for (id, &value) in values.iter().enumerate() {
                        let valid = value % divisor == 0 && minimum.is_none_or(|n| value >= n);
                        assert_eq!(allowed(&mask, id), valid, "divisor={divisor}, minimum={minimum:?}, value={value}");
                        if valid {
                            let mut state = instance.start();
                            state.commit_token(id as u32).unwrap();
                            assert!(state.is_accepting());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn integer_multiple_exact_large_divisor_is_bounded_or_explicitly_unsupported() {
    let vocab = Vocab::new(vec![(0, b"0".to_vec()), (1, b"1000003".to_vec()), (2, b"7".to_vec())]);
    let unbounded = Grammar::from_json_schema(r#"{"type":"integer","multipleOf":1000003}"#);
    assert!(unbounded.compile_with(&vocab, BuildOptions::default().optimization(Optimization::FastBuild)).is_err());
    // Finite exact expansion remains supported even above the generic DFA budget.
    let bounded = Grammar::from_json_schema(r#"{"type":"integer","multipleOf":1000003,"minimum":0,"maximum":2000006}"#)
        .compile_with(&vocab, BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
    for id in [0, 1] {
        let mut state = bounded.start();
        state.commit_token(id).unwrap();
        assert!(state.is_accepting());
    }
    let mut state = bounded.start();
    if state.commit_token(2).is_ok() {
        assert!(!state.is_accepting());
    }
}
