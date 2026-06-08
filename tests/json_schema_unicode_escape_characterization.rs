use std::{env, ffi::OsString};

use glrmask::{Constraint, Vocab};

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
        if key == "GLRMASK_LLGUIDANCE_COMPAT" {
            let enabled = value != "0" && !value.is_empty();
            glrmask::set_test_compat_mode(enabled);
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
        if self.key == "GLRMASK_LLGUIDANCE_COMPAT" {
            let original_enabled = self.original.as_ref().is_some_and(|value| value != "0" && !value.is_empty());
            glrmask::set_test_compat_mode(original_enabled);
        }
    }
}

fn vocab(entries: &[&[u8]]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect(),
        None,
    )
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

fn assert_token_allowed_after_prefix(schema: &str, prefix: &[u8], token: &[u8]) {
    let _compat = EnvVarGuard::set("GLRMASK_LLGUIDANCE_COMPAT", "1");
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();
    let mask = state.mask();
    assert!(token_allowed(&mask, 0), "expected token {:?} to be allowed after prefix {:?}", token, prefix);
}

fn assert_token_rejected_after_prefix(schema: &str, prefix: &[u8], token: &[u8]) {
    let _compat = EnvVarGuard::set("GLRMASK_LLGUIDANCE_COMPAT", "1");
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();
    let mask = state.mask();
    assert!(
        !token_allowed(&mask, 0),
        "expected token {:?} to be rejected after prefix {:?}",
        token,
        prefix,
    );
}

#[test]
fn o60309_original_unicode_prefix_is_allowed_at_custom_envs_start() {
    let schema = r#"{
        "type": "string",
        "pattern": "^(KONG_\\w+=\\S+)*(\\sKONG_\\w+=\\S+)*$"
    }"#;
    assert_token_allowed_after_prefix(schema, br#"""#, br#"\u"#);
}

#[test]
fn o60309_space_backslash_token_is_rejected_at_custom_envs_start() {
    let schema = r#"{
        "type": "string",
        "pattern": "^(KONG_\\w+=\\S+)*(\\sKONG_\\w+=\\S+)*$"
    }"#;
    assert_token_rejected_after_prefix(schema, br#"""#, b" \\");
}

#[test]
fn o82657_original_unicode_prefix_is_allowed_for_css_class_pattern() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[\\w\\s-]+$"
    }"#;
    assert_token_allowed_after_prefix(schema, br#"""#, br#"\u"#);
}

#[test]
fn o82657_quote_backslash_token_is_rejected_at_etag_value_start() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "etag": {
                "type": "string",
                "pattern": "^[\\w\\.:-]+$"
            }
        },
        "required": ["etag"],
        "additionalProperties": true
    }"#;
    assert_token_rejected_after_prefix(schema, br#"{"etag":"#, b" \"\\");
}

#[test]
fn o71827_original_unicode_prefix_is_allowed_for_category_id_pattern() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[^A-Z_ ]+$"
    }"#;
    assert_token_allowed_after_prefix(schema, br#"""#, br#"\u"#);
}

#[test]
fn o71827_partial_unicode_hex_token_is_rejected_for_category_id_pattern() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[^A-Z_ ]+$"
    }"#;
    assert_token_rejected_after_prefix(schema, br#"""#, br#"\uB"#);
}

#[test]
fn o21175_guid_backslash_token_is_rejected_at_string_start() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$"
    }"#;
    assert_token_rejected_after_prefix(schema, br#"""#, br#"\"#);
}