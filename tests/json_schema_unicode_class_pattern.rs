use glrmask::{Constraint, Vocab};

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

    let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
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

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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
    let schema = r#"{
        "type": "string",
        "pattern": "^\\S+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u00B2"#.to_vec()),
        ]);

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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
    let schema = r#"{
        "type": "string",
        "pattern": "^[\\w\\s-]+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u00A0"#.to_vec()),
        ]);

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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
    let schema = r#"{
        "type": "string",
        "pattern": "^[^A-Z_ ]+$"
    }"#;

    let vocab = Vocab::new(
        vec![
            (0, br#"\u"#.to_vec()),
            (1, br#"\u00B2"#.to_vec()),
        ]);

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
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
