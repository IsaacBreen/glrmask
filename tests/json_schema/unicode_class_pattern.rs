use glrmask::{Constraint as Constraint, Vocab};
use glrmask_json_schema::__private::string::{JsonStringCompatMode, TEST_COMPAT_MODE};

/// Panic-safe RAII pin for the thread-local JSON-string compat mode (the live
/// control for pattern lowering). Mirrors the root-lib `CompatModeGuard`
/// pattern: saves the current mode, sets the requested one, and restores the
/// original on drop. Thread-local only — no environment mutation, so parallel
/// libtest threads cannot observe each other's pins.
struct CompatModeGuard {
    original: JsonStringCompatMode,
}

impl CompatModeGuard {
    fn json_schema() -> Self {
        Self::set(JsonStringCompatMode::JsonSchema)
    }

    fn native() -> Self {
        Self::set(JsonStringCompatMode::LlGuidanceNative)
    }

    fn set(mode: JsonStringCompatMode) -> Self {
        let original = TEST_COMPAT_MODE.with(|cell| cell.get());
        TEST_COMPAT_MODE.with(|cell| cell.set(mode));
        Self { original }
    }
}

impl Drop for CompatModeGuard {
    fn drop(&mut self) {
        TEST_COMPAT_MODE.with(|cell| cell.set(self.original));
    }
}

#[test]
fn unicode_class_pattern_preserves_ascii_and_non_ascii_entries() {
    let schema = format!(
        r#"{{
        "type": "string",
        "pattern": "^[A{}]$"
    }}"#,
        '\u{0800}'
    );

    let mut entries = Vec::new();
    for byte in 0..=255u8 {
        entries.push((byte as u32, vec![byte]));
    }
    let vocab = Vocab::new(entries);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(&schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let is_e0_allowed = (mask[224 / 32] >> (224 % 32)) & 1 != 0;
    let is_a_allowed = (mask[65 / 32] >> (65 % 32)) & 1 != 0;

    assert!(is_a_allowed, "glrmask should allow ASCII 'A'");
    assert!(is_e0_allowed, "glrmask should allow the non-ASCII lead byte 0xe0");

    let mut state_token_a = state.clone();
    state_token_a.commit_token(65).unwrap();

    let mut state_token_e0 = state.clone();
    state_token_e0.commit_token(224).unwrap();

    let mut state_bytes_a = state.clone();
    state_bytes_a.commit_bytes(b"A").unwrap();

    let mut state_bytes_e0 = state.clone();
    state_bytes_e0.commit_bytes(&[224]).unwrap();
}

#[test]
fn unicode_class_pattern_preserves_explicit_superscript_two_in_mixed_class() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[A\u00b2]$"
    }"#;

    let mut entries = Vec::new();
    for byte in 0..=255u8 {
        entries.push((byte as u32, vec![byte]));
    }
    let vocab = Vocab::new(entries);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let is_a_allowed = (mask[65 / 32] >> (65 % 32)) & 1 != 0;
    let is_c2_allowed = (mask[194 / 32] >> (194 % 32)) & 1 != 0;

    assert!(is_a_allowed, "glrmask should allow ASCII 'A'");
    assert!(
        is_c2_allowed,
        "glrmask should allow the superscript-two lead byte 0xc2 in a mixed class"
    );

    let mut state_token_a = state.clone();
    state_token_a.commit_token(65).unwrap();

    let mut state_token_c2 = state.clone();
    state_token_c2.commit_token(194).unwrap();

    let mut state_bytes_a = state.clone();
    state_bytes_a.commit_bytes(b"A").unwrap();

    let mut state_bytes_c2 = state.clone();
    state_bytes_c2.commit_bytes(&[194]).unwrap();
}

#[test]
fn unicode_class_pattern_keeps_generic_digit_shorthand_ascii_only() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[A\\d]$"
    }"#;

    let mut entries = Vec::new();
    for byte in 0..=255u8 {
        entries.push((byte as u32, vec![byte]));
    }
    let vocab = Vocab::new(entries);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let is_a_allowed = (mask[65 / 32] >> (65 % 32)) & 1 != 0;
    let is_5_allowed = (mask[53 / 32] >> (53 % 32)) & 1 != 0;
    let is_d9_allowed = (mask[217 / 32] >> (217 % 32)) & 1 != 0;

    assert!(is_a_allowed, "glrmask should allow ASCII 'A'");
    assert!(is_5_allowed, "glrmask should allow ASCII digit '5' from generic \\d");
    assert!(
        !is_d9_allowed,
        "glrmask should not allow the Arabic-Indic zero lead byte 0xd9 from generic \\d"
    );

    let mut state_token_a = state.clone();
    state_token_a.commit_token(65).unwrap();

    let mut state_token_5 = state.clone();
    state_token_5.commit_token(53).unwrap();

    let mut state_bytes_a = state.clone();
    state_bytes_a.commit_bytes(b"A").unwrap();

    let mut state_bytes_5 = state.clone();
    state_bytes_5.commit_bytes(b"5").unwrap();
}

#[test]
fn unicode_class_pattern_keeps_generic_word_shorthand_ascii_only_inside_class() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[A\\w]$"
    }"#;

    let mut entries = Vec::new();
    for byte in 0..=255u8 {
        entries.push((byte as u32, vec![byte]));
    }
    let vocab = Vocab::new(entries);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let is_a_allowed = (mask[65 / 32] >> (65 % 32)) & 1 != 0;
    let is_5_allowed = (mask[53 / 32] >> (53 % 32)) & 1 != 0;
    let is_underscore_allowed = (mask[95 / 32] >> (95 % 32)) & 1 != 0;
    let is_d8_allowed = (mask[216 / 32] >> (216 % 32)) & 1 != 0;

    assert!(is_a_allowed, "glrmask should allow ASCII 'A'");
    assert!(is_5_allowed, "glrmask should allow ASCII digit '5' from generic \\w");
    assert!(
        is_underscore_allowed,
        "glrmask should allow ASCII underscore from generic \\w"
    );
    assert!(
        !is_d8_allowed,
        "glrmask should not allow non-ASCII lead byte 0xd8 from generic \\w"
    );
}

#[test]
fn unicode_class_pattern_keeps_generic_word_shorthand_ascii_only_outside_class() {
    let schema = r#"{
        "type": "string",
        "pattern": "^\\w+$"
    }"#;

    let mut entries = Vec::new();
    for byte in 0..=255u8 {
        entries.push((byte as u32, vec![byte]));
    }
    let vocab = Vocab::new(entries);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let is_a_allowed = (mask[65 / 32] >> (65 % 32)) & 1 != 0;
    let is_5_allowed = (mask[53 / 32] >> (53 % 32)) & 1 != 0;
    let is_underscore_allowed = (mask[95 / 32] >> (95 % 32)) & 1 != 0;
    let is_eb_allowed = (mask[235 / 32] >> (235 % 32)) & 1 != 0;

    assert!(is_a_allowed, "glrmask should allow ASCII letters from generic \\w");
    assert!(is_5_allowed, "glrmask should allow ASCII digits from generic \\w");
    assert!(
        is_underscore_allowed,
        "glrmask should allow ASCII underscore from generic \\w"
    );
    assert!(
        !is_eb_allowed,
        "glrmask should not allow non-ASCII lead byte 0xeb from generic \\w"
    );
}

#[test]
fn unicode_class_pattern_preserves_explicit_fullwidth_digit_range_in_mixed_class() {
    let schema = r#"{
        "type": "string",
        "pattern": "^[A\uff10-\uff19]$"
    }"#;

    let mut entries = Vec::new();
    for byte in 0..=255u8 {
        entries.push((byte as u32, vec![byte]));
    }
    let vocab = Vocab::new(entries);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let is_a_allowed = (mask[65 / 32] >> (65 % 32)) & 1 != 0;
    let is_ef_allowed = (mask[239 / 32] >> (239 % 32)) & 1 != 0;

    assert!(is_a_allowed, "glrmask should allow ASCII 'A'");
    assert!(
        is_ef_allowed,
        "glrmask should allow the fullwidth-digit lead byte 0xef in a mixed class"
    );

    let mut state_token_a = state.clone();
    state_token_a.commit_token(65).unwrap();

    let mut state_token_ef = state.clone();
    state_token_ef.commit_token(239).unwrap();

    let mut state_bytes_a = state.clone();
    state_bytes_a.commit_bytes(b"A").unwrap();

    let mut state_bytes_ef = state.clone();
    state_bytes_ef.commit_bytes(&[239]).unwrap();
}

#[test]
fn non_whitespace_class_accepts_unicode_escape_prefix_and_valid_bmp_escape() {
    // Legacy full-JSON pin: these BMP `\uXXXX` escape assertions assume the
    // historical full JSON lexical language (all `\uXXXX` spellings). The
    // integration-binary default is llguidance-native, which omits non-ASCII
    // BMP escape spellings in value patterns.
    let _compat = CompatModeGuard::json_schema();
    let schema = r#"{
        "type": "string",
        "pattern": "^\\S+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u00B2"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let allows_prefix = (mask[0 / 32] >> (0 % 32)) & 1 != 0;
    let allows_escape = (mask[1 / 32] >> (1 % 32)) & 1 != 0;

    assert!(allows_prefix, "glrmask should allow bare \\u when some non-whitespace BMP escapes remain valid");
    assert!(allows_escape, "glrmask should allow a valid non-whitespace BMP unicode escape");

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u00B2"#).unwrap();
}

#[test]
fn mixed_word_whitespace_class_accepts_bmp_unicode_escape_for_nbsp() {
    // Legacy full-JSON pin: see non_whitespace_class_... above. NBSP escape
    // `\u00A0` is only spellable under the full JSON lexical language.
    let _compat = CompatModeGuard::json_schema();
    let schema = r#"{
        "type": "string",
        "pattern": "^[\\w\\s-]+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u00A0"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let allows_prefix = (mask[0 / 32] >> (0 % 32)) & 1 != 0;
    let allows_escape = (mask[1 / 32] >> (1 % 32)) & 1 != 0;

    assert!(allows_prefix, "glrmask should allow bare \\u when some whitespace BMP escapes remain valid");
    assert!(allows_escape, "glrmask should allow a valid NBSP unicode escape for \\s");

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u00A0"#).unwrap();
}

#[test]
fn negated_ascii_class_accepts_unicode_escape_prefix_and_valid_bmp_escape() {
    // Legacy full-JSON pin: see non_whitespace_class_... above. The ² escape
    // `\u00B2` is only spellable under the full JSON lexical language.
    let _compat = CompatModeGuard::json_schema();
    let schema = r#"{
        "type": "string",
        "pattern": "^[^A-Z_ ]+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u00B2"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let allows_prefix = (mask[0 / 32] >> (0 % 32)) & 1 != 0;
    let allows_escape = (mask[1 / 32] >> (1 % 32)) & 1 != 0;

    assert!(allows_prefix, "glrmask should allow bare \\u when some negated-class BMP escapes remain valid");
    assert!(allows_escape, "glrmask should allow a valid BMP unicode escape in the negated class");

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u00B2"#).unwrap();
}

#[test]
fn literal_prefix_pattern_accepts_unicode_escape_for_printable_bmp_character() {
    let schema = r#"{
        "type": "string",
        "pattern": "^KONG_$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u004BONG_"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    let allows_prefix = (mask[0 / 32] >> (0 % 32)) & 1 != 0;
    let allows_escape = (mask[1 / 32] >> (1 % 32)) & 1 != 0;

    assert!(allows_prefix, "glrmask should allow bare \\u when a printable literal can be spelled via unicode escape");
    assert!(allows_escape, "glrmask should allow a printable literal via unicode escape spelling");

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u004BONG_"#).unwrap();
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

#[test]
fn non_whitespace_class_native_mode_accepts_canonical_and_control_escape_but_rejects_bmp_escape() {
    // LlGuidance-native counterpart of
    // `non_whitespace_class_accepts_unicode_escape_prefix_and_valid_bmp_escape`.
    // Native value patterns omit non-ASCII BMP `\uXXXX` spellings, but keep
    // canonical raw spellings and the `\u0000`-`\u001F` control escapes.
    let _compat = CompatModeGuard::native();
    let schema = r#"{
        "type": "string",
        "pattern": "^\\S+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, b"A".to_vec()),
            (1, b" ".to_vec()),
            (2, br#"\u"#.to_vec()),
            (3, br#"\u00B2"#.to_vec()),
            (4, br#"\u0001"#.to_vec()),
            (5, "\u{00b2}".as_bytes().to_vec()),
            (6, br#"\u0020"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), "native \\S should allow canonical 'A'");
    assert!(!token_allowed(&mask, 1), "native \\S should reject a raw space");
    assert!(token_allowed(&mask, 2), "native \\S should allow bare \\u while a control escape remains valid");
    assert!(!token_allowed(&mask, 3), "native \\S should reject the non-ASCII BMP escape \\u00B2");
    assert!(token_allowed(&mask, 4), "native \\S should allow the control escape \\u0001");
    assert!(token_allowed(&mask, 5), "native \\S should allow raw superscript-two bytes");
    assert!(!token_allowed(&mask, 6), "native \\S should reject the whitespace escape \\u0020");

    let mut committed = state.clone();
    committed.commit_bytes(b"A").unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u0001"#).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes("\u{00b2}".as_bytes()).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut rejected = state.clone();
    assert!(rejected.commit_bytes(br#"\u00B2"#).is_err(), "native \\S should not commit \\u00B2");

    let mut rejected = state.clone();
    assert!(rejected.commit_bytes(br#"\u0020"#).is_err(), "native \\S should not commit \\u0020");
}

#[test]
fn mixed_word_whitespace_class_native_mode_accepts_canonical_and_tab_escape_but_rejects_nbsp_escape() {
    // LlGuidance-native counterpart of
    // `mixed_word_whitespace_class_accepts_bmp_unicode_escape_for_nbsp`.
    // The NBSP escape `\u00A0` is not in the native value language, while the
    // raw NBSP bytes and the tab escape `\u0009` are.
    let _compat = CompatModeGuard::native();
    let schema = r#"{
        "type": "string",
        "pattern": "^[\\w\\s-]+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, b"A".to_vec()),
            (1, b" ".to_vec()),
            (2, b"-".to_vec()),
            (3, br#"\u"#.to_vec()),
            (4, br#"\u00A0"#.to_vec()),
            (5, br#"\u0009"#.to_vec()),
            (6, "\u{00a0}".as_bytes().to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), "native [\\w\\s-] should allow canonical 'A'");
    assert!(token_allowed(&mask, 1), "native [\\w\\s-] should allow a raw space");
    assert!(token_allowed(&mask, 2), "native [\\w\\s-] should allow canonical '-'");
    assert!(token_allowed(&mask, 3), "native [\\w\\s-] should allow bare \\u while a whitespace escape remains valid");
    assert!(!token_allowed(&mask, 4), "native [\\w\\s-] should reject the NBSP escape \\u00A0");
    assert!(token_allowed(&mask, 5), "native [\\w\\s-] should allow the tab escape \\u0009");
    assert!(token_allowed(&mask, 6), "native [\\w\\s-] should allow raw NBSP bytes");

    let mut committed = state.clone();
    committed.commit_bytes(b"A").unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes(b" ").unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u0009"#).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes("\u{00a0}".as_bytes()).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut rejected = state.clone();
    assert!(rejected.commit_bytes(br#"\u00A0"#).is_err(), "native [\\w\\s-] should not commit \\u00A0");
}

#[test]
fn negated_ascii_class_native_mode_enforces_negation_across_escape_spellings() {
    // LlGuidance-native counterpart of
    // `negated_ascii_class_accepts_unicode_escape_prefix_and_valid_bmp_escape`.
    // The negation must hold for escape spellings too: excluded characters'
    // escapes are rejected while an allowed character's escape commits.
    let _compat = CompatModeGuard::native();
    let schema = r#"{
        "type": "string",
        "pattern": "^[^A-Z_ ]+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"A".to_vec()),
            (2, b" ".to_vec()),
            (3, br#"\u"#.to_vec()),
            (4, br#"\u00B2"#.to_vec()),
            (5, br#"\u0001"#.to_vec()),
            (6, br#"\u0041"#.to_vec()),
            (7, br#"\u0020"#.to_vec()),
            (8, br#"\u0061"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), "native negated class should allow canonical 'a'");
    assert!(!token_allowed(&mask, 1), "native negated class should reject canonical 'A'");
    assert!(!token_allowed(&mask, 2), "native negated class should reject a raw space");
    assert!(token_allowed(&mask, 3), "native negated class should allow bare \\u while an escape remains valid");
    assert!(!token_allowed(&mask, 4), "native negated class should reject the non-ASCII BMP escape \\u00B2");
    assert!(token_allowed(&mask, 5), "native negated class should allow the control escape \\u0001");
    assert!(!token_allowed(&mask, 6), "native negated class should reject the excluded escape \\u0041");
    assert!(!token_allowed(&mask, 7), "native negated class should reject the excluded escape \\u0020");
    assert!(token_allowed(&mask, 8), "native negated class should allow the permitted escape \\u0061");

    let mut committed = state.clone();
    committed.commit_bytes(b"a").unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u0001"#).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes(br#"\u0061"#).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut rejected = state.clone();
    assert!(rejected.commit_bytes(br#"\u00B2"#).is_err(), "native negated class should not commit \\u00B2");

    let mut rejected = state.clone();
    assert!(rejected.commit_bytes(br#"\u0041"#).is_err(), "native negated class should not commit \\u0041");
}

#[test]
fn mixed_ascii_non_ascii_class_native_mode_accepts_raw_superscript_two_but_rejects_escape() {
    // LlGuidance-native counterpart of
    // `unicode_class_pattern_preserves_explicit_superscript_two_in_mixed_class`.
    // The raw ² spelling stays canonical; the `\u00B2` spelling is not in the
    // native value language, so even the bare `\u` prefix is rejected here.
    let _compat = CompatModeGuard::native();
    let schema = r#"{
        "type": "string",
        "pattern": "^[A\u00b2]$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, b"A".to_vec()),
            (1, "\u{00b2}".as_bytes().to_vec()),
            (2, br#"\u00B2"#.to_vec()),
            (3, br#"\u"#.to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"").unwrap();

    let mask = state.mask();
    assert!(token_allowed(&mask, 0), "native mixed class should allow canonical 'A'");
    assert!(token_allowed(&mask, 1), "native mixed class should allow raw superscript-two bytes");
    assert!(!token_allowed(&mask, 2), "native mixed class should reject the escape \\u00B2");
    assert!(!token_allowed(&mask, 3), "native mixed class should reject bare \\u when no escape remains valid");

    let mut committed = state.clone();
    committed.commit_bytes(b"A").unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut committed = state.clone();
    committed.commit_bytes("\u{00b2}".as_bytes()).unwrap();
    committed.commit_bytes(b"\"").unwrap();

    let mut rejected = state.clone();
    assert!(rejected.commit_bytes(br#"\u00B2"#).is_err(), "native mixed class should not commit \\u00B2");
}
