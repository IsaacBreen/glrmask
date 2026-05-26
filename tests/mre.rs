use std::{env, ffi::OsString, sync::Mutex};

use glrmask::{Constraint, Vocab};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::set_var(key, value);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => unsafe {
                env::set_var(self.key, value);
            },
            None => unsafe {
                env::remove_var(self.key);
            },
        }
    }
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

fn total_parser_stack_count(state: &glrmask::ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

fn total_final_stack_count(stacks: &[(u32, Vec<Vec<u32>>)]) -> usize {
    stacks.iter().map(|(_, stacks)| stacks.len()).sum()
}

fn byte_vocab_with_separator_token() -> (Vocab, u32) {
    let mut entries: Vec<(u32, Vec<u8>)> = (0u32..=255).map(|byte| (byte, vec![byte as u8])).collect();
    let separator_token_id = 256;
    entries.push((separator_token_id, b" \"".to_vec()));
    (Vocab::new(entries, None), separator_token_id)
}

#[test]
fn chunk16_bounded_service_name_allows_spaces_token_after_open_quote() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _chunk = EnvVarGuard::set("GLRMASK_STRING_REPEAT_CHUNK", "16");

    let schema = r#"{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "serviceName": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100
            }
        },
        "required": ["serviceName"]
    }"#;
    let prefix = br#"{"serviceName": ""#;
    let vocab = Vocab::new(vec![(0, vec![b' '; 24])], None);

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let table_ambiguities = constraint.table_ambiguous_actions();
    assert!(
        table_ambiguities.is_empty(),
        "table-level ambiguity should be eliminated before runtime: {table_ambiguities:#?}",
    );
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();

    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
}

#[test]
fn minimized_sp343_separator_wave_matches_profile_oracle() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _trace = EnvVarGuard::set("GLRMASK_PROFILE_ADVANCE_TRACE", "1");

    let schema = r#"{
        "properties": {
            "failure": {
                "properties": {
                    "messages": {
                        "items": {
                            "anyOf": [
                                {
                                    "additionalProperties": false,
                                    "properties": {
                                        "error": { "type": "string" },
                                        "field": {}
                                    },
                                    "required": ["field"]
                                },
                                {
                                    "additionalProperties": false,
                                    "properties": {
                                        "error": {},
                                        "schemaKey": {}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    }"#;
    let prefix = b"{\"failure\": {\"messages\": [{\"error\": \"Json parsing issue\",";
    let (vocab, separator_token_id) = byte_vocab_with_separator_token();

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();
    assert!(
        !state.has_parser_ambiguity(),
        "parser ambiguity should be eliminated before runtime stack fanout: {:?}",
        state.debug_parser_stacks(),
    );

    assert!(token_allowed(&state.mask(), separator_token_id as usize));
    assert_eq!(state.parser_path_count(1_000_000), 1);
    assert_eq!(total_parser_stack_count(&state), 1);

    let (advances, final_stacks, commit_profile) = state.commit_token_per_advance(separator_token_id).unwrap();

    assert_eq!(advances.len(), 1);
    assert_eq!(commit_profile.n_advances, 1);
    assert_eq!(commit_profile.adv_n_nondet_waves, 0);
    assert_eq!(commit_profile.adv_n_nondet_reduce_ops, 0);
    assert_eq!(commit_profile.adv_n_nondet_merges, 0);
    assert_eq!(commit_profile.adv_n_nondet_isolates, 0);

    let advance = &advances[0];
    assert_eq!(advance.profile.n_nondet_waves, 0);
    assert_eq!(advance.profile.n_nondet_reduce_ops, 0);
    assert_eq!(advance.profile.n_nondet_merges, 0);
    assert_eq!(advance.profile.n_nondet_isolates, 0);
    assert_eq!(advance.gss_stacks_before.len(), 1);
    assert_eq!(advance.gss_stacks_after.len(), 1);
    assert_eq!(total_final_stack_count(&final_stacks), 1);

    let trace = advance.profile.trace.as_ref().expect("advance trace should be enabled for this test");
    assert!(trace.nondet_waves.is_empty(), "separator advance should stay deterministic: {trace:#?}");
}


#[test]
fn recognition_quotient_preserves_masks_and_commit_results_against_legacy_table() {
    let _lock = ENV_LOCK.lock().unwrap();

    let schema = r#"{
        "properties": {
            "failure": {
                "properties": {
                    "messages": {
                        "items": {
                            "anyOf": [
                                {
                                    "additionalProperties": false,
                                    "properties": {
                                        "error": { "type": "string" },
                                        "field": {}
                                    },
                                    "required": ["field"]
                                },
                                {
                                    "additionalProperties": false,
                                    "properties": {
                                        "error": {},
                                        "schemaKey": {}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    }"#;
    let prefixes: &[&[u8]] = &[
        b"",
        b"{\"failure\": ",
        b"{\"failure\": {\"messages\": [{\"error\": \"Json parsing issue\",",
    ];
    let probe_tokens = [b'{' as u32, b'"' as u32, b' ' as u32, b',' as u32, 256u32];

    let (vocab, _) = byte_vocab_with_separator_token();
    let legacy = {
        let _disabled = EnvVarGuard::set("GLRMASK_DISABLE_RECOGNITION_QUOTIENT", "1");
        Constraint::from_json_schema(schema, &vocab).unwrap()
    };
    let optimized = Constraint::from_json_schema(schema, &vocab).unwrap();

    for &prefix in prefixes {
        let mut legacy_state = legacy.start();
        let mut optimized_state = optimized.start();
        let legacy_result = legacy_state.commit_bytes(prefix);
        let optimized_result = optimized_state.commit_bytes(prefix);
        assert_eq!(legacy_result.is_ok(), optimized_result.is_ok(), "prefix {prefix:?}");
        if legacy_result.is_err() {
            continue;
        }

        assert_eq!(legacy_state.mask(), optimized_state.mask(), "mask at prefix {prefix:?}");
        for token in probe_tokens {
            let mut legacy_after = legacy_state.clone();
            let mut optimized_after = optimized_state.clone();
            let legacy_commit = legacy_after.commit_token(token);
            let optimized_commit = optimized_after.commit_token(token);
            assert_eq!(
                legacy_commit.is_ok(),
                optimized_commit.is_ok(),
                "token {token} after prefix {prefix:?}",
            );
            if legacy_commit.is_ok() {
                assert_eq!(
                    legacy_after.mask(),
                    optimized_after.mask(),
                    "post-commit mask for token {token} after prefix {prefix:?}",
                );
            }
        }
    }
}
