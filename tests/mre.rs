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
        "recognizer-equivalent parser branches should be represented by one suffix state: {:?}",
        state.debug_parser_stacks(),
    );

    assert!(token_allowed(&state.mask(), separator_token_id as usize));
    assert_eq!(state.parser_path_count(1_000_000), 1);
    assert_eq!(total_parser_stack_count(&state), 1);

    let (advances, final_stacks, commit_profile) = state.commit_token_per_advance(separator_token_id).unwrap();

    assert_eq!(advances.len(), 1);
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
}

#[test]
fn sp343_delete_only_subset_separator_wave_matches_cfa_oracle() {
    let _lock = ENV_LOCK.lock().unwrap();

    // Deletion-only subset of CFA `jsb/data/Snowplow---sp_343_Normalized`.
    // This keeps the same `failure.messages[*]` branches that produce the
    // full-case separator ambiguity, and deletes unrelated root properties,
    // descriptions, and the unused first-branch `value` property. No branch is
    // invented from scratch.
    let schema = r#"{"additionalProperties":false,"properties":{"failure":{"additionalProperties":false,"properties":{"messages":{"items":{"anyOf":[{"additionalProperties":false,"properties":{"error":{"type":"string"},"field":{"maxLength":64,"type":"string"}},"required":["field","error"],"type":"object"},{"additionalProperties":false,"properties":{"error":{"anyOf":[{"additionalProperties":false,"properties":{"error":{"enum":["ResolutionError"]},"lookupHistory":{"items":{"properties":{"attempts":{"minimum":0,"type":"integer"},"errors":{"items":{"properties":{"error":{"enum":["RepoFailure","ClientFailure","NotFound"]},"message":{"maxLength":256,"type":"string"}},"required":["error"],"type":"object"},"minItems":1,"type":"array"},"lastAttempt":{"_format":"date-time","type":"string"},"repostitory":{"type":"string"}},"required":["repository","errors","attempts","lastAttempt"],"type":"object"},"minItems":1,"type":"array"}},"required":["error","lookupHistory"]},{"additionalProperties":false,"properties":{"dataReports":{"items":{"additionalProperties":false,"properties":{"keyword":{"type":["string","null"]},"message":{"type":"string"},"path":{"type":["string","null"]},"targets":{"type":["array","null"]}},"required":["path","message","keyword","targets"]},"minItems":1,"type":"array"},"error":{"enum":["ValidationError"]}},"required":["dataReports"]},{"additionalProperties":false,"properties":{"error":{"enum":["ValidationError"]},"schemaIssues":{"items":{"additionalProperties":false,"properties":{"message":{"type":"string"},"path":{"type":"string"}},"required":["path","message"],"type":"object"},"minItems":1,"type":"array"}},"required":["error"]}]},"schemaKey":{"type":"string"}},"type":"object"}]},"type":"array"}},"required":["messages"],"type":"object"}},"required":["failure"],"type":"object"}"#;
    let prefix = b"{\"failure\": {\"messages\": [{\"error\": \"Json parsing issue\",";
    let (vocab, separator_token_id) = byte_vocab_with_separator_token();

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();

    assert!(token_allowed(&state.mask(), separator_token_id as usize));
    assert!(
        !state.has_parser_ambiguity(),
        "delete-only Snowplow subset should collapse like the minimized MRE: {:?}",
        state.debug_parser_stacks(),
    );

    let (advances, final_stacks, commit_profile) =
        state.commit_token_per_advance(separator_token_id).unwrap();
    assert_eq!(advances.len(), 1);
    assert_eq!(commit_profile.adv_n_nondet_waves, 0);
    assert_eq!(commit_profile.adv_n_nondet_reduce_ops, 0);
    assert_eq!(commit_profile.adv_n_nondet_merges, 0);
    assert_eq!(commit_profile.adv_n_nondet_isolates, 0);
    assert_eq!(total_final_stack_count(&final_stacks), 1);
}
