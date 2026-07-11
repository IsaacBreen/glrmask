//! End-to-end smoke tests for grammar/schema construction, masks, commits, and
//! serialization. Narrow regressions live in dedicated test files.

use std::{
    env,
    ffi::OsString,
    sync::{Mutex, RwLock},
};

use glrmask::{Constraint, ConstraintState, Vocab};

static URI_ENV_LOCK: Mutex<()> = Mutex::new(());
static TI_ENV_LOCK: RwLock<()> = RwLock::new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn unset(key: &'static str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::remove_var(key);
        }
        if key == "GLRMASK_LLGUIDANCE_COMPAT" {
            glrmask::set_test_compat_mode(false);
        }
        Self { key, original }
    }

    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::set_var(key, value);
        }
        if key == "GLRMASK_LLGUIDANCE_COMPAT" {
            let enabled = value != "0" && !value.is_empty();
            glrmask::set_test_compat_mode(enabled);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        let original_enabled = match &self.original {
            Some(value) => unsafe {
                env::set_var(self.key, value);
                let val = value.to_string_lossy();
                val != "0" && !val.is_empty()
            },
            None => unsafe {
                env::remove_var(self.key);
                false
            },
        };
        if self.key == "GLRMASK_LLGUIDANCE_COMPAT" {
            glrmask::set_test_compat_mode(original_enabled);
        }
    }
}

fn vocab(entries: &[&str]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
            .collect(),
        None,
    )
}

fn bytes_vocab() -> Vocab {
    Vocab::new((0u8..=255).map(|b| (b as u32, vec![b])).collect(), None)
}

fn with_stable_ti_env<T>(f: impl FnOnce() -> T) -> T {
    let _guard = TI_ENV_LOCK.read().expect("TI env lock poisoned");
    f()
}

fn ebnf(entries: &[&str], grammar: &str) -> Constraint {
    with_stable_ti_env(|| Constraint::from_ebnf(grammar, &vocab(entries)).unwrap())
}

fn lark_unlocked(entries: &[&str], grammar: &str) -> Constraint {
    Constraint::from_lark(grammar, &vocab(entries)).unwrap()
}

fn lark(entries: &[&str], grammar: &str) -> Constraint {
    with_stable_ti_env(|| lark_unlocked(entries, grammar))
}

fn schema(entries: &[&str], schema: &str) -> Constraint {
    with_stable_ti_env(|| Constraint::from_json_schema(schema, &vocab(entries)).unwrap())
}

fn byte_schema(schema: &str) -> Constraint {
    with_stable_ti_env(|| Constraint::from_json_schema(schema, &bytes_vocab()).unwrap())
}

fn allowed(mask: &[u32]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .flat_map(|(word, &bits)| {
            (0..32).filter_map(move |bit| {
                ((bits >> bit) & 1 != 0).then_some(word * 32 + bit as usize)
            })
        })
        .collect()
}

fn assert_allowed(state: &ConstraintState<'_>, expected: &[usize]) {
    assert_eq!(allowed(&state.mask()), expected);
}

fn commit_tokens(state: &mut ConstraintState<'_>, tokens: &[u32]) {
    for &token in tokens {
        state.commit_token(token).unwrap();
    }
}

fn assert_accepts_tokens(constraint: &Constraint, tokens: &[u32]) {
    let mut state = constraint.start();
    commit_tokens(&mut state, tokens);
    assert!(state.is_finished());
}

fn assert_rejects_token(constraint: &Constraint, prefix: &[u32], token: u32) {
    let mut state = constraint.start();
    commit_tokens(&mut state, prefix);
    assert!(state.commit_token(token).is_err());
}

fn assert_accepts_bytes(constraint: &Constraint, bytes: &[u8]) {
    let mut state = constraint.start();
    state.commit_bytes(bytes).unwrap();
    assert!(state.is_finished());
}

fn assert_rejects_bytes(constraint: &Constraint, bytes: &[u8]) {
    let mut state = constraint.start();
    assert!(state.commit_bytes(bytes).is_err());
}

fn max_paths_and_stacks(constraint: &Constraint, text: &str) -> (usize, usize) {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1_000_000);
    let mut max_stacks = stack_count(&state);

    for &byte in text.as_bytes() {
        state.commit_bytes(&[byte]).unwrap();
        max_paths = max_paths.max(state.parser_path_count(1_000_000));
        max_stacks = max_stacks.max(stack_count(&state));
    }

    (max_paths, max_stacks)
}

fn stack_count(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

#[test]
fn ebnf_masks_and_commits() {
    let constraint = ebnf(&["a", "b", "c"], r#"start ::= "a" ("b" | "c")"#);
    let mut state = constraint.start();
    assert_allowed(&state, &[0]);

    state.commit_token(0).unwrap();
    assert_allowed(&state, &[1, 2]);

    state.commit_token(2).unwrap();
    assert!(state.is_finished());
    assert_rejects_token(&constraint, &[0], 0);
}

#[test]
fn ebnf_repetition_and_optional_separator() {
    let constraint = ebnf(
        &["x", ",", ";"],
        r#"start ::= "x" ("," "x")* ";"?"#,
    );

    assert_accepts_tokens(&constraint, &[0]);
    assert_accepts_tokens(&constraint, &[0, 1, 0, 1, 0]);
    assert_accepts_tokens(&constraint, &[0, 1, 0, 2]);
    assert_rejects_token(&constraint, &[0, 1], 2);
}

#[test]
fn lark_literals_choices_and_terminals() {
    let constraint = lark(
        &["a", "b", "."],
        r#"
        start: ITEM "."
        ITEM: "a" | "b"
        "#,
    );

    let mut state = constraint.start();
    assert_allowed(&state, &[0, 1]);
    state.commit_token(1).unwrap();
    assert_allowed(&state, &[2]);
    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}

#[test]
fn lark_rejects_parser_refs_inside_terminals() {
    let result = with_stable_ti_env(|| {
        Constraint::from_lark(
            r#"
            start: A
            A: inner
            inner: "a"
            "#,
            &vocab(&["a"]),
        )
    });
    assert!(result.is_err());
}

#[test]
fn json_schema_scalar_and_enum() {
    let scalar = schema(&["true", "false"], r#"{"type":"boolean"}"#);
    assert_accepts_tokens(&scalar, &[0]);
    assert_accepts_tokens(&scalar, &[1]);

    let enum_schema = schema(&[r#""red""#, r#""blue""#, r#""green""#], r#"{"enum":["red","blue"]}"#);
    assert_accepts_tokens(&enum_schema, &[0]);
    assert_accepts_tokens(&enum_schema, &[1]);
    assert_rejects_token(&enum_schema, &[], 2);
}

#[test]
fn json_schema_filtered_enum_does_not_overaccept_merged_group() {
    let enum_schema = schema(
        &[r#""a""#, r#""bb""#],
        r#"{"enum":["a","bb"],"minLength":2}"#,
    );
    assert_rejects_token(&enum_schema, &[], 0);
    assert_accepts_tokens(&enum_schema, &[1]);
}

#[test]
fn json_schema_closed_object_all_optional_accepts_sparse_in_order_members() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x"}"#);
    assert_accepts_bytes(&constraint, br#"{"c": "z"}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x", "c": "z"}"#);
    assert_rejects_bytes(&constraint, br#"{"c": "z", "a": "x"}"#);
}

#[test]
fn json_schema_closed_object_min_properties_one_requires_a_property() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "minProperties": 1,
            "additionalProperties": false
        }"#,
    );

    assert_rejects_bytes(&constraint, br#"{}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x"}"#);
    assert_accepts_bytes(&constraint, br#"{"b": "y"}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x", "b": "y"}"#);
}

#[test]
fn json_schema_closed_object_required_property_still_mandatory() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a"],
            "additionalProperties": false
        }"#,
    );

    assert_rejects_bytes(&constraint, br#"{}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x"}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x", "b": "y"}"#);
}

#[test]
fn json_schema_anyof_required_property_factoring_preserves_acceptance() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "boolean"},
                "b": {"type": "boolean"},
                "c": {"type": "boolean"}
            },
            "additionalProperties": false,
            "anyOf": [
                {"required": ["a"]},
                {"required": ["b"]}
            ]
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"a": true}"#);
    assert_accepts_bytes(&constraint, br#"{"b": false}"#);
    assert_accepts_bytes(&constraint, br#"{"a": true, "b": false}"#);
    assert_rejects_bytes(&constraint, br#"{}"#);
    assert_rejects_bytes(&constraint, br#"{"c": true}"#);
    assert_rejects_bytes(&constraint, br#"{"b": false, "a": true}"#);
}

#[test]
fn json_schema_anyof_closed_object_variants_preserve_acceptance() {
    let constraint = byte_schema(
        r#"{
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "boolean"}
                    },
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "boolean"},
                        "x": {"type": "boolean"}
                    },
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "boolean"},
                        "y": {"type": "boolean"}
                    },
                    "additionalProperties": false
                }
            ]
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{}"#);
    assert_accepts_bytes(&constraint, br#"{"a": true}"#);
    assert_accepts_bytes(&constraint, br#"{"x": true}"#);
    assert_accepts_bytes(&constraint, br#"{"a": true, "x": false}"#);
    assert_accepts_bytes(&constraint, br#"{"y": true}"#);
    assert_rejects_bytes(&constraint, br#"{"x": true, "y": false}"#);
    assert_rejects_bytes(&constraint, br#"{"z": true}"#);
}

#[test]
fn json_schema_open_object_all_optional_fixed_props_accepts_tail_only_after_prefix() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "additionalProperties": {"type": "string"}
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x"}"#);
    assert_accepts_bytes(&constraint, br#"{"z": "extra"}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x", "z": "extra", "y": "more"}"#);
}

#[test]
fn json_schema_open_object_rejects_additional_property_before_later_fixed_property() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "additionalProperties": {"type": "string"}
        }"#,
    );

    assert_rejects_bytes(&constraint, br#"{"z": "extra", "b": "y"}"#);
}

#[test]
fn json_schema_open_object_tail_rejects_fixed_property_name() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"}
            },
            "additionalProperties": {"type": "string"}
        }"#,
    );

    assert_rejects_bytes(&constraint, br#"{"a": "x", "a": "again"}"#);
}

#[test]
fn json_schema_large_closed_object_rejects_duplicate_keys_and_trailing_commas() {
    let property_entries = (0..64)
        .map(|index| format!(r#""k{index}": {{"type": "boolean"}}"#))
        .collect::<Vec<_>>();
    let schema_text = format!(
        r#"{{
            "type": "object",
            "properties": {{ {} }},
            "additionalProperties": false
        }}"#,
        property_entries.join(", ")
    );
    let constraint = byte_schema(&schema_text);

    let valid_body = (0..64)
        .map(|index| format!(r#""k{index}": true"#))
        .collect::<Vec<_>>()
        .join(", ");
    let valid = format!("{{{valid_body}}}");
    let duplicate = format!("{{{valid_body}, \"k0\": true}}");
    let trailing_comma = format!("{{{valid_body},}}");

    assert_accepts_bytes(&constraint, valid.as_bytes());
    assert_rejects_bytes(&constraint, duplicate.as_bytes());
    assert_rejects_bytes(&constraint, trailing_comma.as_bytes());
}

#[test]
fn json_schema_optional_label_with_additional_tail_reaches_multiple_gss_paths() {
    let constraint = schema(
        &[r#"{"label": "#, r#""x","#, r#" "id": "#, "1", "}"],
        r#"{
            "type": "object",
            "properties": {
                "label": {"type": "string"}
            },
            "additionalProperties": true
        }"#,
    );
    let mut state = constraint.start();

    state.commit_token(0).unwrap();
    assert_eq!(state.parser_path_count(1_000_000), 1);
    state.commit_token(1).unwrap();
    assert_eq!(state.parser_path_count(1_000_000), 1);
    state.commit_token(2).unwrap();
    assert_eq!(state.parser_path_count(1_000_000), 1);
    state.commit_token(3).unwrap();

    let stacks = state.debug_parser_stacks();
    assert_eq!(state.parser_path_count(1_000_000), 2, "{stacks:?}");
}

#[test]
fn json_schema_open_object_required_fixed_property_remains_mandatory_with_ap_tail() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        }"#,
    );

    assert_rejects_bytes(&constraint, br#"{"z": "extra"}"#);
    assert_accepts_bytes(&constraint, br#"{"a": "x", "z": "extra"}"#);
}

#[test]
fn json_schema_rejects_invalid_utf8_in_string() {
    let constraint = byte_schema(r#"{"type":"string"}"#);
    let mut state = constraint.start();
    state.commit_bytes(&[b'"']).unwrap();
    assert!(state.commit_bytes(&[0xff]).is_err());
}

#[test]
fn json_schema_uri_format_default_mode_accepts_basic_uri() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _uri_mode = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_URI_MODE");

    let constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    let mut state = constraint.start();
    state.commit_bytes(br#""https://example.com""#).unwrap();
    assert!(state.is_finished());
}

#[test]
fn json_schema_uri_format_structured_mode_accepts_basic_uri() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _uri_mode = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "structured");

    let constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    let mut state = constraint.start();
    state.commit_bytes(br#""https://example.com""#).unwrap();
    assert!(state.is_finished());
}

#[test]
fn json_schema_uri_format_approx_mode_rejects_bracketed_host() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _approx = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "approx");

    let approx_constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    assert_rejects_bytes(&approx_constraint, br#""http://[not::strict::]/path""#);
}

#[test]
fn json_schema_uri_format_default_rejects_bracketed_host() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _uri_mode = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_URI_MODE");

    let strict_constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    assert_rejects_bytes(&strict_constraint, br#""http://[not::strict::]/path""#);
}

#[test]
fn json_schema_uri_format_approx_mode_rejects_non_uri_string() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _approx = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "approx");

    let constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    assert_rejects_bytes(&constraint, br#""not a uri""#);
}

#[test]
fn json_schema_uri_format_default_rejects_missing_scheme() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _uri_mode = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_URI_MODE");

    let constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    assert_rejects_bytes(&constraint, br#""not a uri""#);
}

#[test]
fn json_schema_uri_format_default_rejects_relative_path() {
    let _lock = URI_ENV_LOCK.lock().unwrap();
    let _uri_mode = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_URI_MODE");

    let constraint = byte_schema(r#"{"type":"string","format":"uri"}"#);
    assert_rejects_bytes(&constraint, br#""/not-absolute""#);
}

#[test]
fn json_schema_email_format_accepts_basic_email() {
    let constraint = byte_schema(r#"{"type":"string","format":"email"}"#);
    assert_accepts_bytes(&constraint, br#""john.doe@example.com""#);
}

#[test]
fn json_schema_date_format_accepts_valid_date() {
    let constraint = byte_schema(r#"{"type":"string","format":"date"}"#);
    assert_accepts_bytes(&constraint, br#""2021-01-15""#);
}

#[test]
fn json_schema_date_format_rejects_zero_month() {
    let constraint = byte_schema(r#"{"type":"string","format":"date"}"#);
    assert_rejects_bytes(&constraint, br#""2021-00-15""#);
}

#[test]
fn json_schema_date_format_rejects_thirteenth_month() {
    let constraint = byte_schema(r#"{"type":"string","format":"date"}"#);
    assert_rejects_bytes(&constraint, br#""2021-13-15""#);
}

#[test]
fn json_schema_date_format_rejects_zero_day() {
    let constraint = byte_schema(r#"{"type":"string","format":"date"}"#);
    assert_rejects_bytes(&constraint, br#""2021-12-00""#);
}

#[test]
fn json_schema_date_format_rejects_day_thirty_two() {
    let constraint = byte_schema(r#"{"type":"string","format":"date"}"#);
    assert_rejects_bytes(&constraint, br#""2021-12-32""#);
}

#[test]
fn json_schema_email_format_rejects_empty_string() {
    let constraint = byte_schema(r#"{"type":"string","format":"email"}"#);
    assert_rejects_bytes(&constraint, br#""""#);
}

#[test]
fn json_schema_email_format_rejects_missing_at_sign() {
    let constraint = byte_schema(r#"{"type":"string","format":"email"}"#);
    assert_rejects_bytes(&constraint, br#""not an email""#);
}

#[test]
fn json_schema_number_multiple_of_001_accepts_two_decimal_places() {
    let constraint = byte_schema(r#"{"type":"number","multipleOf":0.01}"#);
    assert_accepts_bytes(&constraint, b"1.23");
}

#[test]
fn json_schema_number_multiple_of_001_rejects_three_decimal_places() {
    let constraint = byte_schema(r#"{"type":"number","multipleOf":0.01}"#);
    let mut state = constraint.start();
    assert!(state.commit_bytes(b"1.234").is_err());
}

#[test]
fn json_schema_number_multiple_of_001_rejects_extra_significant_digits() {
    let constraint = byte_schema(r#"{"type":"number","multipleOf":0.01}"#);
    let mut state = constraint.start();
    assert!(state.commit_bytes(b"1.001").is_err());
}

#[test]
fn json_schema_number_multiple_of_001_rejects_trailing_zero_spelling() {
    let constraint = byte_schema(r#"{"type":"number","multipleOf":0.01}"#);
    assert_rejects_bytes(&constraint, b"1.230");
}

#[test]
fn json_schema_number_multiple_of_05_accepts_half_steps() {
    let constraint = byte_schema(r#"{"type":"number","multipleOf":0.5}"#);
    assert_accepts_bytes(&constraint, b"1.5");
}

#[test]
fn json_schema_number_multiple_of_05_rejects_non_half_steps() {
    let constraint = byte_schema(r#"{"type":"number","multipleOf":0.5}"#);
    let mut state = constraint.start();
    assert!(state.commit_bytes(b"1.2").is_err());
}

#[test]
fn json_schema_number_multiple_of_10_accepts_integer_multiple() {
    let constraint = byte_schema(
        r#"{"type":"array","items":{"type":"number","multipleOf":10},"minItems":1,"maxItems":1}"#,
    );
    assert_accepts_bytes(&constraint, b"[20]");
}

#[test]
fn json_schema_number_multiple_of_10_rejects_non_multiple() {
    let constraint = byte_schema(
        r#"{"type":"array","items":{"type":"number","multipleOf":10},"minItems":1,"maxItems":1}"#,
    );
    let mut state = constraint.start();
    assert!(state.commit_bytes(b"[21]").is_err());
}

#[test]
fn json_schema_pattern_with_max_length_token_mask_rejects_overlong_identifier() {
    let schema_text = r#"{
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 18,
                    "pattern": "^[A-Za-z][A-Za-z0-9_]*"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        }"#;
    let token_constraint = schema(
        &["OptionsItemSelected", "TemperatureSensor"],
        schema_text,
    );

    let mut token_state = token_constraint.start();
    token_state.commit_bytes(br#"{"name": ""#).unwrap();
    assert_eq!(allowed(&token_state.mask()), vec![1]);

    let mut overlong_token_state = token_constraint.start();
    overlong_token_state.commit_bytes(br#"{"name": ""#).unwrap();
    assert!(overlong_token_state.commit_token(0).is_err());

    let mut allowed_token_state = token_constraint.start();
    allowed_token_state.commit_bytes(br#"{"name": ""#).unwrap();
    allowed_token_state.commit_token(1).unwrap();

    let mut token_prefix_state = token_constraint.start();
    token_prefix_state.commit_bytes(br#"{"name": ""#).unwrap();
    assert!(token_prefix_state.commit_bytes(b"OptionsItemSelected").is_err());

    let constraint = byte_schema(
        schema_text,
    );

    let mut prefix_state = constraint.start();
    prefix_state.commit_bytes(br#"{"name": "#).unwrap();
    assert!(prefix_state.commit_bytes(b"OptionsItemSelected").is_err());

    let mut state = constraint.start();
    assert!(state
        .commit_bytes(br#"{"name": "OptionsItemSelected"}"#)
        .is_err());
}

#[test]
fn json_schema_pattern_with_s_separator_accepts_tab_and_formfeed_prefixes() {
    let constraint = byte_schema(
        r#"{
            "type": "string",
            "pattern": "^(KONG_\\w+=\\S+)*(\\sKONG_\\w+=\\S+)*$"
        }"#,
    );

    let mut tab_state = constraint.start();
    tab_state.commit_bytes(br#"""#).unwrap();
    tab_state.commit_bytes(br#"\t"#).unwrap();

    let mut formfeed_state = constraint.start();
    formfeed_state.commit_bytes(br#"""#).unwrap();
    formfeed_state.commit_bytes(br#"\f"#).unwrap();

    assert_accepts_bytes(&constraint, br#""\tKONG_A=x""#);
    assert_accepts_bytes(&constraint, br#""\fKONG_A=x""#);
}

#[test]
fn json_schema_bounded_free_text_pattern_rejects_leading_space_slash_token() {
    let constraint = byte_schema(
        r#"{
            "type": "string",
            "maxLength": 200,
            "minLength": 0,
            "pattern": "^$|(^(?:\\S+\\s+){0,19}\\S+$)"
        }"#,
    );

    assert_accepts_bytes(&constraint, br#""REST API""#);
    assert_rejects_bytes(&constraint, br#"" /""#);
}

#[test]
fn json_schema_optional_decimal_pattern_rejects_backslash_prefix_token() {
    let constraint = byte_schema(
        r#"{
            "type": "string",
            "pattern": "^$|^\\d{1,15}(?:\\.\\d{1,5})?$"
        }"#,
    );

    assert_accepts_bytes(&constraint, br#""""#);
    assert_accepts_bytes(&constraint, br#""123.45""#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#""\\"#).is_err());
}

#[test]
fn json_schema_pattern_accepts_decoded_quote_string() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^\"$"}"#);
    assert_accepts_bytes(&constraint, br#""\"""#);

    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();
    assert!(state.commit_bytes(b"\"").is_err());
}

#[test]
fn json_schema_pattern_accepts_decoded_backslash_string() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^\\\\$"}"#);
    assert_accepts_bytes(&constraint, br#""\\""#);
}

#[test]
fn json_schema_pattern_accepts_decoded_backslash_incrementally() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^[\\\\]+$"}"#);
    let mut state = constraint.start();

    state.commit_bytes(b"\"").unwrap();
    state.commit_bytes(b"\\").unwrap();
    state.commit_bytes(b"\\").unwrap();
    assert!(allowed(&state.mask()).contains(&(b'"' as usize)));
    state.commit_bytes(b"\"").unwrap();
    assert!(state.is_finished());
}

#[test]
fn json_schema_pattern_accepts_decoded_backslash_fused_token() {
    let constraint = schema(
        &["\"", "\\\\\"", "\\\\", "\\u"],
        r#"{"type":"string","pattern":"^[\\\\]+$"}"#,
    );
    let mut state = constraint.start();

    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask()).contains(&1));
    state.commit_token(1).unwrap();
    assert!(state.is_finished());
}

#[test]
fn json_schema_pattern_range_accepts_backslash_fused_token() {
    let constraint = schema(
        &["\"", "\\\\\"", "\\\\", "\\u"],
        r#"{"type":"string","pattern":"^[.-_]+$"}"#,
    );
    let mut state = constraint.start();

    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask()).contains(&1));
    state.commit_token(1).unwrap();
    assert!(state.is_finished());
}

#[test]
fn json_schema_pattern_accepts_decoded_newline_escape_spellings() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^\\n$"}"#);
    assert_accepts_bytes(&constraint, br#""\n""#);
    let mut state = constraint.start();
    assert!(state.commit_bytes(br#""\u000A""#).is_err());
}

#[test]
fn json_schema_pattern_accepts_multi_digit_dimensions_string() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^[0-9]+x[0-9]+$"}"#);
    assert_accepts_bytes(&constraint, br#""1920x1080""#);
}

#[test]
fn json_schema_pattern_rejects_extra_separator_in_dimensions_string() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^[0-9]+x[0-9]+$"}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#""1920x108x0""#).is_err());
}

#[test]
fn json_schema_pattern_dimensions_prefix_rejects_second_separator_token() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^[0-9]+x[0-9]+$"}"#);
    let mut state = constraint.start();

    state.commit_bytes(br#""1920x108"#).unwrap();
    let mask = allowed(&state.mask());
    assert!(mask.contains(&(b'0' as usize)));
    assert!(!mask.contains(&(b'x' as usize)));
    assert!(state.commit_bytes(b"x").is_err());

    let mut ok_state = constraint.start();
    ok_state.commit_bytes(br#""1920x108"#).unwrap();
    ok_state.commit_bytes(b"0").unwrap();
}

#[test]
fn json_schema_dot_pattern_rejects_invalid_utf8_bytes() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^.$"}"#);
    assert_accepts_bytes(&constraint, br#""a""#);

    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();
    assert!(state.commit_bytes(&[0xff]).is_err());
}

#[test]
fn json_schema_suffix_dot_pattern_allows_valid_utf8_lead_byte_token() {
    let constraint = byte_schema(r#"{"type":"string","pattern":"^.*.txt$"}"#);

    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();
    assert!(allowed(&state.mask()).contains(&0xd3));
}

#[test]
fn json_schema_pattern_properties_accepts_encoded_quote_key() {
    let constraint = byte_schema(
        r#"{"type":"object","patternProperties":{"^\"$":{"type":"integer"}},"additionalProperties":false}"#,
    );
    assert_accepts_bytes(&constraint, br#"{"\"": 1}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"": 1}"#).is_err());
}

#[test]
fn json_schema_pattern_properties_match_decoded_fixed_quote_key() {
    let constraint = byte_schema(
        r#"{
            "type":"object",
            "properties":{"\"":{"type":"integer"}},
            "patternProperties":{"^\"$":{"minimum":2,"maximum":5}},
            "additionalProperties":false
        }"#,
    );
    assert_accepts_bytes(&constraint, br#"{"\"": 2}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"\"": 1}"#).is_err());
}

#[test]
fn json_schema_pattern_properties_do_not_reaccept_declared_key() {
    let constraint = byte_schema(
        r#"{
            "type":"object",
            "properties":{"kind":{"const":"event"}},
            "patternProperties":{"^.*$":{"type":"integer"}},
            "additionalProperties":false
        }"#,
    );
    let mut event_state = constraint.start();
    assert!(event_state.commit_bytes(br#"{"kind": "event"}"#).is_err());

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"kind": 1}"#).is_err());
}

#[test]
fn json_schema_additional_properties_exclude_pattern_matched_keys() {
    let constraint = byte_schema(
        r#"{
            "type":"object",
            "patternProperties":{"^x":{"type":"string"}},
            "additionalProperties":{"type":"integer"}
        }"#,
    );
    assert_accepts_bytes(&constraint, br#"{"x1": "ok"}"#);
    assert_accepts_bytes(&constraint, br#"{"y": 1}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"x1": 1}"#).is_err());
}

#[test]
fn json_schema_additional_property_pattern_addback_does_not_reaccept_declared_key() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "applicationId": {"type": "string"},
                "nested": {
                    "type": "object",
                    "patternProperties": {
                        "^[0-9A-Za-z_-]{1,255}$": {"type": "number"}
                    },
                    "additionalProperties": true
                }
            },
            "additionalProperties": true
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"x_key": 123}"#);
    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"applicationId": 123}"#).is_err());
}

#[test]
fn json_schema_additional_property_literal_addback_restores_globally_excluded_key() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {"known": {"type": "string"}},
                    "additionalProperties": false
                },
                "b": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"}
                }
            },
            "required": ["b"],
            "additionalProperties": false
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"b": {"known": 1}}"#);
}

#[test]
fn json_schema_additional_property_pattern_addback_restores_globally_excluded_pattern() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "patternProperties": {"^x": {"type": "integer"}},
                    "additionalProperties": false
                },
                "b": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["b"],
            "additionalProperties": false
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"b": {"x1": "ok"}}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"b": {"x1": 1}}"#).is_err());
}

#[test]
fn json_schema_open_anyof_variant_additional_properties_allow_other_variant_keys() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"}
                    }
                },
                {
                    "type": "object",
                    "properties": {
                        "b": {"type": "boolean"}
                    }
                }
            ]
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"a": []}"#);
    assert_accepts_bytes(&constraint, br#"{"b": []}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"[]"#).is_err());
}

#[test]
fn json_schema_number_multiple_of_keeps_exclusive_maximum() {
    let constraint = byte_schema(
        r#"{
            "type": "number",
            "multipleOf": 10,
            "exclusiveMaximum": 100
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"20"#);

    let mut too_large = constraint.start();
    assert!(too_large.commit_bytes(br#"600"#).is_err());

    let mut boundary = constraint.start();
    assert!(boundary.commit_bytes(br#"100"#).is_err());
}

#[test]
fn json_schema_object_property_untyped_numeric_assertions_allow_non_numbers() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "size": {"maximum": 100000}
            }
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"size": 50000}"#);
    assert_accepts_bytes(&constraint, br#"{"size": "free"}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"size": 100001}"#).is_err());
}

#[test]
fn json_schema_object_property_untyped_string_assertions_allow_non_strings() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "name": {"pattern": "^.*.txt$"}
            }
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"name": "example.txt"}"#);
    assert_accepts_bytes(&constraint, br#"{"name": []}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"name": "example.csv"}"#).is_err());
}

#[test]
fn json_schema_object_property_untyped_array_assertions_allow_non_arrays() {
    let constraint = byte_schema(
        r#"{
            "type": "object",
            "properties": {
                "dataFormats": {
                    "maxItems": 1,
                    "items": {"enum": ["application/json"]}
                }
            }
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"dataFormats": ["application/json"]}"#);
    assert_accepts_bytes(&constraint, br#"{"dataFormats": true}"#);
    assert_accepts_bytes(&constraint, br#"{"dataFormats": "anything"}"#);

    let mut invalid_item = constraint.start();
    assert!(invalid_item.commit_bytes(br#"{"dataFormats": ["text/plain"]}"#).is_err());

    let mut too_many_items = constraint.start();
    assert!(too_many_items
        .commit_bytes(br#"{"dataFormats": ["application/json", "application/json"]}"#)
        .is_err());

    let mut state = constraint.start();
    state.commit_bytes(br#"{"dataFormats":"#).unwrap();
    state.commit_bytes(b" true").unwrap();
}

#[test]
fn json_schema_additional_property_required_only_key_does_not_fall_back_through_ap() {
    let constraint = byte_schema(
        r#"{
            "type":"object",
            "required":["id"],
            "additionalProperties":{"type":"string"}
        }"#,
    );

    assert_accepts_bytes(&constraint, br#"{"id": "ok"}"#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"{"id": 1}"#).is_err());
}

#[test]
fn json_schema_const_quote_reencodes_literal() {
    let constraint = byte_schema(r#"{"const":"\""}"#);
    assert_accepts_bytes(&constraint, br#""\"""#);

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#""\\""#).is_err());
}

#[test]
fn json_schema_pattern_and_date_format_rejects_invalid_month_token() {
    let constraint = byte_schema(
        r#"{
            "type": "string",
            "format": "date",
            "pattern": "^([0-9]{4})(-([0-9]{2}))?(-([0-9]{2}))?$"
        }"#,
    );

    assert_accepts_bytes(&constraint, br#""2023-09-01""#);
    assert_rejects_bytes(&constraint, br#""2023-93""#);

    let mut state = constraint.start();
    state.commit_bytes(br#""2023-"#).unwrap();
    assert!(state.commit_bytes(b"93").is_err());
}

#[test]
fn nullable_repeat_alternative_accepts_nonempty_branch_before_nullable_suffix() {
    let grammar = r#"
        start s;

        nt s ::= "a" host "c"* "a";
        nt host ::= "d" | "b"*;
    "#;

    let tiny_vocab = vocab(&["a", "b"]);
    let constraint =
        with_stable_ti_env(|| Constraint::from_glrm_grammar(grammar, &tiny_vocab).unwrap());

    let mut empty_host = constraint.start();
    empty_host.commit_bytes(b"aa").unwrap();
    assert!(empty_host.is_finished());

    let mut single_repeat_host = constraint.start();
    single_repeat_host.commit_bytes(b"aba").unwrap();
    assert!(single_repeat_host.is_finished());
}

#[test]
fn commit_bytes_and_commit_tokens_agree() {
    let constraint = ebnf(&["a", "b", "ab"], r#"start ::= "a" "b" | "ab""#);

    let mut by_tokens = constraint.start();
    by_tokens.commit_tokens(&[0, 1]).unwrap();
    assert!(by_tokens.is_finished());

    let mut by_bytes = constraint.start();
    by_bytes.commit_bytes(b"ab").unwrap();
    assert!(by_bytes.is_finished());
}

#[test]
fn force_reports_deterministic_prefix() {
    let constraint = ebnf(&["a", "b", "c"], r#"start ::= "a" "b" ("c")?"#);
    let mut state = constraint.start();
    assert_eq!(state.force(), vec![0, 1, 2]);

    state.commit_tokens(&[0, 1]).unwrap();
    assert!(state.is_finished());
}

#[test]
fn save_load_roundtrip_preserves_behavior() {
    let constraint = ebnf(&["a", "b"], r#"start ::= "a" "b""#);
    let bytes = constraint.save();
    let loaded = Constraint::load(&bytes).unwrap();
    assert_accepts_tokens(&loaded, &[0, 1]);
}

#[test]
fn runtime_payload_v1_roundtrip_preserves_behavior_without_overlay() {
    let constraint = ebnf(&["a", "b"], r#"start ::= "a" "b""#);
    let bytes = constraint.save_runtime_payload_v1();
    let loaded = Constraint::load_runtime_payload_v1(&bytes).unwrap();
    assert_accepts_tokens(&loaded, &[0, 1]);
}

#[test]
fn runtime_payload_v2_roundtrip_preserves_split_parser_overlay() {
    let constraint = lark(
        &["!", "aaa"],
        r#"
            start: "!" | WORD
            WORD: /[a-z]+/
        "#,
    );
    let bytes = constraint.save_runtime_payload_v2();
    let loaded = Constraint::load_runtime_payload_v2(&bytes).unwrap();
    assert_accepts_tokens(&loaded, &[0]);
    assert_accepts_tokens(&loaded, &[1]);
}

#[test]
fn isolated_and_monolithic_lexer_partitions_are_end_to_end_equivalent() {
    let vocab = vocab(&[
        "a", "aa", "b", "c", "ab", "ac", " b", " c", " ", "  ", "ba", "ca",
        "aa b", "aa c",
    ]);
    let isolated_grammar = r#"
        start start;
        ignore WS;
        lexer group ws ::= WS;
        lexer group a ::= A;
        lexer group b ::= B;
        lexer group c ::= C;
        t WS ::= " "+;
        t A ::= "a"+;
        t B ::= "b";
        t C ::= "c";
        nt start ::= A B | A C | B A | C A;
    "#;
    let monolithic_grammar = r#"
        start start;
        ignore WS;
        lexer group all ::= WS, A, B, C;
        t WS ::= " "+;
        t A ::= "a"+;
        t B ::= "b";
        t C ::= "c";
        nt start ::= A B | A C | B A | C A;
    "#;
    let isolated = Constraint::from_glrm_grammar(isolated_grammar, &vocab).unwrap();
    let monolithic = Constraint::from_glrm_grammar(monolithic_grammar, &vocab).unwrap();

    let mut frontier = vec![(isolated.start(), monolithic.start(), Vec::<u32>::new())];
    for depth in 0..=4 {
        let mut next = Vec::new();
        for (isolated_state, monolithic_state, path) in frontier {
            assert_eq!(
                isolated_state.mask(),
                monolithic_state.mask(),
                "mask differed after token path {path:?}",
            );
            assert_eq!(
                isolated_state.is_finished(),
                monolithic_state.is_finished(),
                "completion differed after token path {path:?}",
            );
            if depth == 4 {
                continue;
            }
            for token in allowed(&isolated_state.mask()) {
                let token = token as u32;
                let mut next_isolated = isolated_state.clone();
                let mut next_monolithic = monolithic_state.clone();
                let isolated_result = next_isolated.commit_token(token);
                let monolithic_result = next_monolithic.commit_token(token);
                assert_eq!(
                    isolated_result.is_ok(),
                    monolithic_result.is_ok(),
                    "commit result differed for token {token} after path {path:?}",
                );
                if isolated_result.is_ok() {
                    let mut next_path = path.clone();
                    next_path.push(token);
                    next.push((next_isolated, next_monolithic, next_path));
                }
            }
        }
        frontier = next;
    }

    let loaded = Constraint::load(&isolated.save()).unwrap();
    assert_eq!(loaded.start().mask(), isolated.start().mask());
}

fn assert_partitioned_runtime_matches_dynamic(
    grammar: &str,
    vocab: &Vocab,
    max_depth: usize,
) {
    let partitioned = Constraint::from_glrm_grammar(grammar, vocab).unwrap();

    let mut frontier = vec![(partitioned.start(), Vec::<u32>::new())];
    for depth in 0..=max_depth {
        let mut next = Vec::new();
        for (partitioned_state, path) in frontier {
            let mut partitioned_dynamic = vec![0; partitioned.mask_len()];
            partitioned_state.fill_mask_dynamic(&mut partitioned_dynamic);
            assert_eq!(
                partitioned_state.mask(),
                partitioned_dynamic,
                "partitioned parser-DWA mask differed from direct dynamic traversal after token path {path:?}\ngrammar:\n{grammar}",
            );
            if depth == max_depth {
                continue;
            }

            let mask = partitioned_state.mask();
            for &token in vocab.entries.keys() {
                let word = token as usize / 32;
                let bit = token % 32;
                let expected_allowed = mask
                    .get(word)
                    .is_some_and(|mask_word| mask_word & (1u32 << bit) != 0);
                let mut next_partitioned = partitioned_state.clone();
                let partitioned_result = next_partitioned.commit_token(token);
                assert_eq!(
                    partitioned_result.is_ok(),
                    expected_allowed,
                    "commit result disagreed with mask for token {token} after path {path:?}\ngrammar:\n{grammar}",
                );
                if expected_allowed {
                    let mut next_path = path.clone();
                    next_path.push(token);
                    next.push((next_partitioned, next_path));
                }
            }
        }
        frontier = next;
    }

    let loaded = Constraint::load(&partitioned.save()).unwrap();
    assert_eq!(loaded.start().mask(), partitioned.start().mask());
}

#[test]
fn partitioned_runtime_matches_dynamic_across_lexer_shapes() {
    let vocab = vocab(&[
        "a", "b", "c", "aa", "bb", "cc", "ab", "ac", "ba", "bc", "abc",
        "aab", "abb", "acc", " ", "  ", " a", "a ", " a ", "ab c",
    ]);
    let cases: &[&str] = &[
            r#"
                start start;
                ignore WS;
                lexer group ws ::= WS;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t WS ::= " "+;
                t A ::= "a"+;
                t B ::= "b";
                t C ::= "c";
                nt item ::= A | B | C;
                nt start ::= item item? item?;
            "#,
            r#"
                start start;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t A ::= "a"*;
                t B ::= "ab" | "b";
                t C ::= "c"+;
                nt item ::= A | B | C;
                nt start ::= item item?;
            "#,
            r#"
                start start;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t A ::= "a" | "ab";
                t B ::= "ab" | "b" | "ba";
                t C ::= "abc" | "bc" | "c";
                nt item ::= A | B | C;
                nt start ::= item item? item?;
            "#,
            r#"
                start start;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t A ::= [abc] - "b";
                t B ::= [ab] & [bc];
                t C ::= "b" | "bc";
                nt item ::= A | B | C;
                nt start ::= item item? item?;
            "#,
            r#"
                start start;
                ignore WS;
                lexer group ws ::= WS;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t WS ::= " "*;
                t A ::= /a(?:b|c)*/;
                t B ::= /b+/;
                t C ::= /c+/;
                nt item ::= A | B | C;
                nt start ::= item item?;
            "#,
    ];

    for &grammar in cases {
        assert_partitioned_runtime_matches_dynamic(grammar, &vocab, 3);
    }
}

#[test]
fn monolithic_runtime_matches_dynamic_for_ignore_and_repeated_terminals() {
    let vocab = vocab(&[
        "a", "b", "c", "aa", "bb", "cc", "ab", "ac", "ba", "bc", "abc",
        "aab", "abb", "acc", " ", "  ", " a", "a ", " a ", "ab c",
    ]);
    let grammar = r#"
        start start;
        ignore WS;
        t WS ::= " "+;
        t A ::= "a"+;
        t B ::= "b";
        t C ::= "c";
        nt item ::= A | B | C;
        nt start ::= item item? item?;
    "#;
    assert_partitioned_runtime_matches_dynamic(grammar, &vocab, 3);
}

#[test]
fn monolithic_runtime_matches_dynamic_for_overlapping_residual_terminals() {
    let vocab = vocab(&[
        "a", "b", "c", "aa", "bb", "cc", "ab", "ac", "ba", "bc", "abc",
        "aab", "abb", "acc", " ", "  ", " a", "a ", " a ", "ab c",
    ]);
    let grammar = r#"
        start start;
        t A ::= "a" | "ab";
        t B ::= "ab" | "b" | "ba";
        t C ::= "abc" | "bc" | "c";
        nt item ::= A | B | C;
        nt start ::= item item? item?;
    "#;
    assert_partitioned_runtime_matches_dynamic(grammar, &vocab, 4);
}

#[test]
fn residual_terminal_continuation_survives_across_vocab_tokens() {
    let vocab = vocab(&["a", "b", "c", "ab", "ba", "bc", "abc"]);
    let grammar = r#"
        start start;
        t A ::= "a";
        t B ::= "b" | "ba";
        nt item ::= A | B;
        nt start ::= item item? item?;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();

    state.commit_token(1).unwrap(); // "b"
    state.commit_token(3).unwrap(); // "ab"; bytes "bab" = B("ba") B("b")

    let mask = state.mask();
    assert!(
        mask[3 / 32] & (1u32 << (3 % 32)) != 0,
        "next token ab must be admitted: babab = B(ba) B(ba) B(b)"
    );
}

#[test]
fn partitioned_repeat_continuation_survives_ignore_prefixed_token() {
    let vocab = vocab(&[
        "a", "b", "c", "aa", "bb", "cc", "ab", "ac", "ba", "bc", "abc",
        "aab", "abb", "acc", " ", "  ", " a", "a ", " a ", "ab c",
    ]);
    let grammar = r#"
        start start;
        ignore WS;
        lexer group ws ::= WS;
        lexer group a ::= A;
        lexer group b ::= B;
        lexer group c ::= C;
        t WS ::= " "+;
        t A ::= "a"+;
        t B ::= "b";
        t C ::= "c";
        nt item ::= A | B | C;
        nt start ::= item item? item?;
    "#;
    let monolithic_grammar = r#"
        start start;
        ignore WS;
        lexer group all ::= WS, A, B, C;
        t WS ::= " "+;
        t A ::= "a"+;
        t B ::= "b";
        t C ::= "c";
        nt item ::= A | B | C;
        nt start ::= item item? item?;
    "#;
    let partitioned = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let monolithic = Constraint::from_glrm_grammar(monolithic_grammar, &vocab).unwrap();
    let mut partitioned_state = partitioned.start();
    let mut monolithic_state = monolithic.start();

    for token in [0, 6, 16] {
        partitioned_state.commit_token(token).unwrap();
        monolithic_state.commit_token(token).unwrap();
    }

    assert_eq!(partitioned_state.mask(), monolithic_state.mask());
    assert_allowed(&partitioned_state, &[0, 3, 14, 15, 17]);
    let mut dynamic = vec![0; partitioned.mask_len()];
    partitioned_state.fill_mask_dynamic(&mut dynamic);
    assert_eq!(partitioned_state.mask(), dynamic);
}

#[test]
fn nullable_terminal_root_loop_is_preserved_by_isolated_partitions() {
    let vocab = vocab(&["a", "aa", "b", "ab", "aab"]);
    let isolated = Constraint::from_glrm_grammar(
        r#"
            start start;
            lexer group a ::= A;
            lexer group b ::= B;
            t A ::= "a"*;
            t B ::= "b";
            nt start ::= A B;
        "#,
        &vocab,
    )
    .unwrap();
    let monolithic = Constraint::from_glrm_grammar(
        r#"
            start start;
            lexer group all ::= A, B;
            t A ::= "a"*;
            t B ::= "b";
            nt start ::= A B;
        "#,
        &vocab,
    )
    .unwrap();

    assert_eq!(isolated.start().mask(), monolithic.start().mask());
    for token_path in [&[2][..], &[3], &[4], &[0, 2], &[1, 2]] {
        let mut isolated_state = isolated.start();
        let mut monolithic_state = monolithic.start();
        for &token in token_path {
            isolated_state.commit_token(token).unwrap();
            monolithic_state.commit_token(token).unwrap();
            assert_eq!(isolated_state.mask(), monolithic_state.mask());
            assert_eq!(isolated_state.is_finished(), monolithic_state.is_finished());
        }
        assert!(isolated_state.is_finished(), "path {token_path:?} did not finish");
    }
}

#[test]
fn plan_style_mask_buffer_matches_mask() {
    let constraint = ebnf(&["a", "b"], r#"start ::= "a" "b""#);
    let mut state = constraint.start();
    let mut buffer = vec![0; constraint.mask_len()];

    state.fill_mask(&mut buffer);
    assert_eq!(buffer, state.mask());
    assert_allowed(&state, &[0]);

    state.commit_token(0).unwrap();
    state.fill_mask(&mut buffer);
    assert_eq!(buffer, state.mask());
    assert_allowed(&state, &[1]);
}

#[test]
fn direct_glrm_ordered_suffix_model_has_stack_ambiguity() {
    let grammar = r#"
        start start;
        nt f0 ::= "a";
        nt f1 ::= "b";
        nt f2 ::= "c";
        nt f3 ::= "d";
        nt f4 ::= "e";
        nt f5 ::= "f";
        nt f6 ::= "g";
        nt f7 ::= "h";
        nt f8 ::= "i";
        nt v0 ::= f0 "," f1 ("," f2)? "," f3 "," f4 "," f5 "," f6 ("," f7)? ("," f8)?;
        nt v1 ::= f0 "," f1 ("," f2)? "," f3 "," f4 "," f5 ("," f6)? "," f7 ("," f8)?;
        nt v2 ::= f0 "," f1 ("," f2)? "," f3 "," f4 "," f5 ("," f6)? ("," f7)? "," f8;
        nt start ::= v0 | v1 | v2;
    "#;

    let constraint = with_stable_ti_env(|| {
        Constraint::from_glrm_grammar(grammar, &bytes_vocab()).unwrap()
    });
    let (max_paths, max_stacks) = max_paths_and_stacks(&constraint, "a,b,c,d,e,f,g,h");
    assert_eq!((max_paths, max_stacks), (3, 3));
}


#[test]
fn json_schema_kubernetes_container_ports_prefix_has_single_stack_path() {
    // Minimized from Kubernetes kb_996: this keeps the same two-stack
    // ordered-object/additional-property split shape at an open exact key.
    // The empty property names are intentional minimization artifacts; the
    // required shape is one array property with an item schema ending in
    // open key `g`, plus a later sibling array whose item schema shares the
    // prefix but lacks `g`.
    const K8S_ORDERED_PORTS_SCHEMA_FRAGMENT: &str = r####"
    {
      "properties": {
        "a": {"items": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}, "z": {"type": "string"}}}},
        "b": {"items": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}}
      },
      "additionalProperties": false
    }"####;
    const K8S_ORDERED_PORTS_PREFIX: &[u8] = br####"{"a": [{"x": "", ""####;

    let constraint = with_stable_ti_env(|| {
        Constraint::from_json_schema(K8S_ORDERED_PORTS_SCHEMA_FRAGMENT, &bytes_vocab()).unwrap()
    });
    let mut state = constraint.start();
    state.commit_bytes(K8S_ORDERED_PORTS_PREFIX).unwrap();

    let stacks = state.debug_parser_stacks();
    let mut stack_values = stacks
        .iter()
        .flat_map(|(_, parser_stacks)| parser_stacks.iter())
        .map(|(stack, _)| stack.clone())
        .collect::<Vec<_>>();
    stack_values.sort_unstable();
    stack_values.dedup();
    assert_eq!(stack_values.len(), 1, "{stacks:?}");

    // The old regression shape kept two equivalent stack suffixes here.
    // Partitioned lexers may retain several lexer residuals, but they must all
    // carry the same parser stack rather than reintroducing parser ambiguity.
}

#[test]
fn direct_glrm_minimized_lowered_schema_has_two_stack_split() {
    let grammar = r#"start s;nt k::="a""b"*;nt i::=k"b"?;nt s::="d"i;"#;
    let constraint = with_stable_ti_env(|| {
        Constraint::from_glrm_grammar(grammar, &bytes_vocab()).unwrap()
    });

    let mut state = constraint.start();
    for &byte in b"dab" {
        state.commit_bytes(&[byte]).unwrap();
    }
    let stacks = state.debug_parser_stacks();
    let path_count = state.parser_path_count(10);

    assert_eq!(path_count, 2, "{stacks:?}");
    assert_eq!(stacks.len(), 1, "{stacks:?}");
    assert_eq!(stack_count(&state), 2, "{stacks:?}");

    // This is the minimized GLRM lowering of the JSON-schema split above.
    // `k` models the ordered known-property continuation, while `i` adds the
    // following tail continuation. Both nonterminals are load-bearing for the
    // schema-shaped suffix lengths.
    let stack_values = stacks
        .iter()
        .flat_map(|(_, stacks)| stacks.iter().map(|(stack, _)| stack.clone()))
        .collect::<Vec<_>>();
    assert_eq!(stack_values.len(), 2, "{stacks:?}");
    assert_eq!(stack_values[0][..2], stack_values[1][..2], "{stacks:?}");
    let mut suffix_lengths = [stack_values[0].len() - 2, stack_values[1].len() - 2];
    suffix_lengths.sort_unstable();
    assert_eq!(suffix_lengths, [1, 2], "{stacks:?}");
}

#[test]
fn direct_glrm_minimized_lowered_schema_collapses_when_tail_token_differs() {
    let grammar = r#"start s;nt k::="a""b"*;nt i::=k"c"?;nt s::="d"i;"#;
    let constraint = with_stable_ti_env(|| {
        Constraint::from_glrm_grammar(grammar, &bytes_vocab()).unwrap()
    });

    let mut state = constraint.start();
    for &byte in b"dab" {
        state.commit_bytes(&[byte]).unwrap();
    }
    let stacks = state.debug_parser_stacks();

    // This differs from the split test by one literal: the tail continuation is
    // `"c"?` instead of `"b"?`. The consumed `b` can only be part of `k`'s
    // `"b"*`, so the parser has no competing "known-list vs tail" continuation.
    assert_eq!(state.parser_path_count(10), 1, "{stacks:?}");
    assert_eq!(stacks.len(), 1, "{stacks:?}");
    assert_eq!(stack_count(&state), 1, "{stacks:?}");
}


#[test]
fn terminal_interchangeability_minimal_two_byte_counterexample_matches_baseline() {
    let _lock = TI_ENV_LOCK.write().unwrap();
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _disable_feature = EnvVarGuard::unset("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY");
    let _disable_validation = EnvVarGuard::unset("GLRMASK_VALIDATE_L2P_TERMINAL_INTERCHANGEABILITY");

    // Minimal counterexample to label-only post-DWA reconstruction. A and B
    // share a restricted-DFA interchange map, but B's continuation after A
    // must come from a transported whole-DWA initial copy, not an A-edge clone.
    let entries = ["aa"];
    let grammar = r#"
        start: A B
        A: /a(aaaa)*/
        B: /aaa(aaaa)*/
    "#;
    let baseline = lark_unlocked(&entries, grammar);

    drop(_disable_feature);
    drop(_disable_validation);
    let _enable_feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let _assert_equal = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );
    let expanded = lark_unlocked(&entries, grammar);

    let observe = |constraint: &Constraint, sequence: &[u32]| {
        let mut state = constraint.start();
        for &token in sequence {
            if state.commit_token(token).is_err() {
                return None;
            }
        }
        Some((state.is_finished(), allowed(&state.mask())))
    };
    for first in 0..entries.len() as u32 {
        for second in 0..entries.len() as u32 {
            for sequence in [Vec::new(), vec![first], vec![first, second]] {
                assert_eq!(
                    observe(&baseline, &sequence),
                    observe(&expanded, &sequence),
                    "terminal interchangeability changed token prefix {sequence:?}",
                );
            }
        }
    }

    for prefix_len in 0..=12usize {
        let prefix = vec![b'a'; prefix_len];
        let observe_bytes = |constraint: &Constraint| {
            let mut state = constraint.start();
            if state.commit_bytes(&prefix).is_err() {
                return None;
            }
            Some((state.is_finished(), allowed(&state.mask())))
        };
        assert_eq!(
            observe_bytes(&baseline),
            observe_bytes(&expanded),
            "terminal interchangeability changed byte prefill {prefix:?}",
        );
    }
}

#[test]
fn strict_terminal_interchangeability_reference_matches_baseline_l2p_artifact() {
    let _lock = TI_ENV_LOCK.write().unwrap();
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _disable_feature = EnvVarGuard::unset("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY");
    let _disable_validation = EnvVarGuard::unset("GLRMASK_VALIDATE_L2P_TERMINAL_INTERCHANGEABILITY");

    // Advancing through the `a` cycle exchanges the two terminal residuals.
    // The enabled build also performs its own local terminal-DWA/id-map
    // comparison against this baseline before returning.
    let entries = [
        "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "x",
    ];
    let grammar = r#"
        start: choice choice
        choice: A | B
        A: /a(?:aaaa)*/
        B: /aaa(?:aaaa)*/
    "#;
    let baseline = lark_unlocked(&entries, grammar);

    drop(_disable_feature);
    drop(_disable_validation);
    let _enable_feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let _assert_equal = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );
    let expanded = lark_unlocked(&entries, grammar);

    let observe = |constraint: &Constraint, sequence: &[u32]| {
        let mut state = constraint.start();
        for &token in sequence {
            if state.commit_token(token).is_err() {
                return None;
            }
        }
        Some((state.is_finished(), allowed(&state.mask())))
    };
    for first in 0..entries.len() as u32 {
        for second in 0..entries.len() as u32 {
            for sequence in [Vec::new(), vec![first], vec![first, second]] {
                assert_eq!(
                    observe(&baseline, &sequence),
                    observe(&expanded, &sequence),
                    "strict terminal interchangeability changed token prefix {sequence:?}",
                );
            }
        }
    }

    for prefix_len in 0..=12usize {
        let prefix = vec![b'a'; prefix_len];
        let observe_bytes = |constraint: &Constraint| {
            let mut state = constraint.start();
            if state.commit_bytes(&prefix).is_err() {
                return None;
            }
            Some((state.is_finished(), allowed(&state.mask())))
        };
        assert_eq!(
            observe_bytes(&baseline),
            observe_bytes(&expanded),
            "strict terminal interchangeability changed byte prefill {prefix:?}",
        );
    }
}

#[test]
fn strict_terminal_interchangeability_reference_validates_one_terminal_position() {
    let _lock = TI_ENV_LOCK.write().unwrap();
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let _assert_equal = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );
    let entries = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "x"];
    let grammar = r#"
        start: A | B
        A: /a(?:aaaa)*/
        B: /aaa(?:aaaa)*/
    "#;
    let _ = lark_unlocked(&entries, grammar);
}

#[test]
fn transported_terminal_interchangeability_with_ignore_equals_baseline_artifact() {
    let _lock = TI_ENV_LOCK.write().unwrap();
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let _assert_equal = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );

    let entries = [
        "a", "aa", "aaa", "aaaa", " ", "a ", " aaa", "aaaa ", "x",
    ];
    let grammar = r#"
        start: choice choice
        choice: A | B
        A: /a(?:aaaa)*/
        B: /aaa(?:aaaa)*/
        WS: / +/
        %ignore WS
    "#;
    let _ = lark_unlocked(&entries, grammar);
}

#[test]
fn three_member_terminal_interchangeability_equals_baseline_artifact() {
    let _lock = TI_ENV_LOCK.write().unwrap();
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let _assert_equal = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );

    let entries = ["x", "xx", "xxx", "xxxx", "a"];
    let grammar = r#"
        start: item item item
        item: A | B | C
        A: "x"
        B: "x"
        C: "x"
    "#;
    let _ = lark_unlocked(&entries, grammar);
}

#[test]
fn independent_terminal_interchangeability_classes_equal_baseline_artifact() {
    let _lock = TI_ENV_LOCK.write().unwrap();
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let _assert_equal = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );

    let entries = ["x", "y", "xx", "xy", "yx", "yy", "xxy", "yxx", "z"];
    let grammar = r#"
        start: item item item
        item: A | B | C | D
        A: "x"
        B: "x"
        C: "y"
        D: "y"
    "#;
    let _ = lark_unlocked(&entries, grammar);
}
