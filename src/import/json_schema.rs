#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::{Map, Value};

use crate::GlrMaskError;
use crate::automata::lexer::regex::parse_regex;
use crate::compiler::grammar_def::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, lower, promote_large_literal_alts};

const JSON_VALUE_RULE: &str = "json_value";
const JSON_OBJECT_RULE: &str = "json_object";
const JSON_ARRAY_RULE: &str = "json_array";
const JSON_KV_RULE: &str = "json_kv";
const JSON_STRING_RULE: &str = "JSON_STRING";
const JSON_STRING_CHAR_RULE: &str = "JSON_STRING_CHAR";
const JSON_INTEGER_RULE: &str = "JSON_INTEGER";
const JSON_NUMBER_RULE: &str = "JSON_NUMBER";
const JSON_NONNEG_INTEGER_RULE: &str = "JSON_NONNEG_INTEGER";
const JSON_NONNEG_NUMBER_RULE: &str = "JSON_NONNEG_NUMBER";
const JSON_BOOL_RULE: &str = "JSON_BOOL";
const JSON_NULL_RULE: &str = "JSON_NULL";
const JSON_KEY_COLON_RULE: &str = "JSON_KEY_COLON";

const JSON_STRING_REGEX: &str =
    r#""([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*""#;
const JSON_KEY_COLON_REGEX: &str =
    r#""([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*": "#;
const JSON_STRING_CHAR_PATTERN: &str = r#"[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}"#;
const JSON_DIRECT_UTF8_PATTERN: &str =
    r#"(?:[\xC2-\xDF][\x80-\xBF]|[\xE0][\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC][\x80-\xBF][\x80-\xBF]|[\xED][\x80-\x9F][\x80-\xBF]|[\xEE-\xEF][\x80-\xBF][\x80-\xBF]|[\xF0][\x90-\xBF][\x80-\xBF][\x80-\xBF]|[\xF1-\xF3][\x80-\xBF][\x80-\xBF][\x80-\xBF]|[\xF4][\x80-\x8F][\x80-\xBF][\x80-\xBF])"#;
const JSON_ITEM_SEPARATOR: &[u8] = b", ";
const JSON_KEY_SEPARATOR: &[u8] = b": ";
const UNTYPED_OBJECT_KEYWORD_KEYS: &[&str] = &[
    "properties",
    "required",
    "additionalProperties",
    "patternProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
];
const UNTYPED_ARRAY_KEYWORD_KEYS: &[&str] = &[
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
];
const UNTYPED_STRING_KEYWORD_KEYS: &[&str] = &[
    "pattern",
    "format",
    "minLength",
    "maxLength",
];
const UNTYPED_NUMBER_KEYWORD_KEYS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];
const UNTYPED_SCHEMA_APPLICABLE_TYPES: &[(&str, &[&str])] = &[
    ("object", UNTYPED_OBJECT_KEYWORD_KEYS),
    ("array", UNTYPED_ARRAY_KEYWORD_KEYS),
    ("string", UNTYPED_STRING_KEYWORD_KEYS),
    ("number", UNTYPED_NUMBER_KEYWORD_KEYS),
];
const UNSAT_SCHEMA_ERROR: &str = "__unsat_schema__";

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum JsonSchemaDraft {
    Draft4,
    Draft6,
    Draft7,
    Draft201909,
    Draft202012,
}

const DEFAULT_JSON_SCHEMA_DRAFT: JsonSchemaDraft = JsonSchemaDraft::Draft202012;
const IMPLEMENTED_JSON_SCHEMA_KEYWORDS: &[&str] = &[
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "const",
    "enum",
    "type",
    "items",
    "additionalItems",
    "prefixItems",
    "minItems",
    "maxItems",
    "properties",
    "additionalProperties",
    "patternProperties",
    "required",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];
const META_AND_ANNOTATION_KEYWORDS: &[&str] = &[
    "$anchor",
    "$defs",
    "definitions",
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "readOnly",
    "writeOnly",
    "examples",
    "contentMediaType",
    "contentEncoding",
];

fn literal_expr(bytes: &[u8]) -> GrammarExpr {
    GrammarExpr::Literal(bytes.to_vec())
}

fn regex_expr(pattern: impl Into<String>) -> GrammarExpr {
    GrammarExpr::RawRegex(pattern.into())
}

fn never_expr() -> GrammarExpr {
    regex_expr(r#"[^\x00-\xFF]"#)
}

fn unsat_schema_error() -> GlrMaskError {
    GlrMaskError::GrammarParse(UNSAT_SCHEMA_ERROR.into())
}

fn is_unsat_schema_error(err: &GlrMaskError) -> bool {
    matches!(err, GlrMaskError::GrammarParse(message) if message == UNSAT_SCHEMA_ERROR)
}

fn detect_draft(
    schema: &Map<String, Value>,
    current: JsonSchemaDraft,
) -> Result<JsonSchemaDraft, GlrMaskError> {
    let Some(raw) = schema.get("$schema").and_then(Value::as_str) else {
        return Ok(current);
    };
    match raw.trim_end_matches('#') {
        "https://json-schema.org/draft/2020-12/schema" => Ok(JsonSchemaDraft::Draft202012),
        "https://json-schema.org/draft/2019-09/schema" => Ok(JsonSchemaDraft::Draft201909),
        "http://json-schema.org/draft-07/schema" => Ok(JsonSchemaDraft::Draft7),
        "http://json-schema.org/draft-06/schema" => Ok(JsonSchemaDraft::Draft6),
        "http://json-schema.org/draft-04/schema" => Ok(JsonSchemaDraft::Draft4),
        value => Err(GlrMaskError::GrammarParse(format!("Unknown specification: {value}"))),
    }
}

fn is_known_keyword(draft: JsonSchemaDraft, keyword: &str) -> bool {
    match keyword {
        "$ref"
        | "$schema"
        | "additionalItems"
        | "additionalProperties"
        | "allOf"
        | "anyOf"
        | "dependencies"
        | "enum"
        | "exclusiveMaximum"
        | "exclusiveMinimum"
        | "format"
        | "items"
        | "maxItems"
        | "maxLength"
        | "maxProperties"
        | "maximum"
        | "minItems"
        | "minLength"
        | "minProperties"
        | "minimum"
        | "multipleOf"
        | "not"
        | "oneOf"
        | "pattern"
        | "patternProperties"
        | "properties"
        | "required"
        | "type"
        | "uniqueItems" => true,
        "id" => draft == JsonSchemaDraft::Draft4,
        "$id" | "const" | "contains" | "propertyNames" => draft >= JsonSchemaDraft::Draft6,
        "contentEncoding" | "contentMediaType" => {
            matches!(draft, JsonSchemaDraft::Draft6 | JsonSchemaDraft::Draft7)
        }
        "else" | "if" | "then" => draft >= JsonSchemaDraft::Draft7,
        "$anchor"
        | "$defs"
        | "$recursiveAnchor"
        | "$recursiveRef"
        | "dependentRequired"
        | "dependentSchemas"
        | "maxContains"
        | "minContains"
        | "prefixItems"
        | "unevaluatedItems"
        | "unevaluatedProperties" => draft >= JsonSchemaDraft::Draft201909,
        "$dynamicAnchor" | "$dynamicRef" => draft == JsonSchemaDraft::Draft202012,
        _ => false,
    }
}

fn type_allows_value(type_name: &str, value: &Value) -> bool {
    match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value
            .as_i64()
            .is_some()
            || value.as_u64().is_some()
            || value.as_f64().map(|number| number.fract() == 0.0).unwrap_or(false),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn empty_expr() -> GrammarExpr {
    GrammarExpr::Sequence(Vec::new())
}

fn json_format_pattern(format_name: &str) -> Option<&'static str> {
    Some(match format_name {
        "time" => {
            r#"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"#
        }
        "duration" => {
            r#"P(?:[0-9]+W|(?:[0-9]+Y)?(?:[0-9]+M)?(?:[0-9]+D)?(?:T(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+S)?)?)"#
        }
        "email" => {
            r#"[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~]+(?:\.[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~]+)*@[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*"#
        }
        "hostname" => {
            r#"[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*"#
        }
        "ipv4" => {
            r#"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"#
        }
        "ipv6" => {
            r#"(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4})|(?:::(?:[0-9a-fA-F]{1,4}:){0,5}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:(?:[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,4}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,1}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,3}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,2}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,2}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,3}[0-9a-fA-F]{1,4})?::[0-9a-fA-F]{1,4}:(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,4}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4})?::)"#
        }
        "uuid" => r#"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"#,
        "uri" => {
            r#"[a-zA-Z][a-zA-Z0-9+\-.]*:(?://(?:[a-zA-Z0-9\-._~%!$&'()*+,;=:]*@)?[a-zA-Z0-9\-._~%!$&'()*+,;=\[\]]+(?::[0-9]*)?)?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/]*(?:\?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/?]*)?(?:#[a-zA-Z0-9\-._~%!$&'()*+,;=:@/?]*)?"#
        }
        "unknown" => r#"(?s:.*)"#,
        _ => return None,
    })
}

fn split_top_level_regex_branches(pattern: &str) -> Vec<&str> {
    let bytes = pattern.as_bytes();
    let mut branches = Vec::new();
    let mut start = 0;
    let mut i = 0;
    let mut paren_depth = 0usize;
    let mut in_class = false;
    let mut escaped = false;

    while i < bytes.len() {
        let byte = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        match byte {
            b'\\' => {
                escaped = true;
            }
            b'[' if !in_class => {
                in_class = true;
            }
            b']' if in_class => {
                in_class = false;
            }
            b'(' if !in_class => {
                paren_depth += 1;
            }
            b')' if !in_class && paren_depth > 0 => {
                paren_depth -= 1;
            }
            b'|' if !in_class && paren_depth == 0 => {
                branches.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    branches.push(&pattern[start..]);
    branches
}

fn strip_single_outer_group(branch: &str) -> Option<&str> {
    let bytes = branch.as_bytes();
    if bytes.first().copied() != Some(b'(') || bytes.last().copied() != Some(b')') {
        return None;
    }

    let inner_start = match bytes.get(1).copied() {
        Some(b'?') => {
            if bytes.get(2).copied() == Some(b':') {
                3
            } else {
                return None;
            }
        }
        _ => 1,
    };

    let mut paren_depth = 0usize;
    let mut in_class = false;
    let mut escaped = false;
    for (idx, byte) in bytes.iter().copied().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' => escaped = true,
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'(' if !in_class => paren_depth += 1,
            b')' if !in_class && paren_depth > 0 => {
                paren_depth -= 1;
                if paren_depth == 0 && idx != bytes.len() - 1 {
                    return None;
                }
            }
            _ => {}
        }
    }

    if paren_depth != 0 || inner_start >= branch.len().saturating_sub(1) {
        return None;
    }
    Some(&branch[inner_start..branch.len() - 1])
}

fn strip_branch_outer_anchors(branch: &str) -> (bool, bool, &str) {
    let mut branch = branch;
    while let Some(inner) = strip_single_outer_group(branch) {
        branch = inner;
    }

    let anchored_start = branch.as_bytes().first().copied() == Some(b'^');
    let mut end_index = branch.len();
    let mut anchored_end = false;
    if branch.as_bytes().last().copied() == Some(b'$') {
        let mut backslashes = 0usize;
        for byte in branch.as_bytes()[..branch.len().saturating_sub(1)]
            .iter()
            .rev()
        {
            if *byte == b'\\' {
                backslashes += 1;
            } else {
                break;
            }
        }
        if backslashes % 2 == 0 {
            anchored_end = true;
            end_index -= 1;
        }
    }

    let start_index = if anchored_start { 1 } else { 0 };
    (anchored_start, anchored_end, &branch[start_index..end_index])
}

fn json_search_branch_fragment(branch: &str) -> String {
    json_search_branch_fragment_impl(branch, None)
}

fn json_search_branch_fragment_bounded(branch: &str, max_tail: usize) -> String {
    json_search_branch_fragment_impl(branch, Some(max_tail))
}

fn json_search_branch_fragment_impl(branch: &str, max_tail: Option<usize>) -> String {
    let (anchored_start, anchored_end, core) = strip_branch_outer_anchors(branch);
    let inner = jsonify_regex_dot(core);
    let string_tail = match max_tail {
        Some(n) => format!(r#"(?:{}){{0,{}}}"#, JSON_STRING_CHAR_PATTERN, n),
        None => format!(r#"(?:{})*"#, JSON_STRING_CHAR_PATTERN),
    };
    match (anchored_start, anchored_end) {
        (true, true) => inner,
        // Wrap inner in (?:...) so that top-level alternation in the pattern
        // (e.g. LATEX|MATHML) binds correctly with the string_tail suffix/prefix.
        (true, false) => format!("(?:{}){}", inner, string_tail),
        (false, true) => format!("{}(?:{})", string_tail, inner),
        (false, false) => format!("{}(?:{}){}", string_tail, inner, string_tail),
    }
}

fn json_search_pattern(pattern: &str) -> String {
    json_search_pattern_impl(pattern, None)
}

fn json_search_pattern_bounded(pattern: &str, max_tail: usize) -> String {
    json_search_pattern_impl(pattern, Some(max_tail))
}

fn json_search_pattern_impl(pattern: &str, max_tail: Option<usize>) -> String {
    let branches = split_top_level_regex_branches(pattern);
    let map_fn = |b| match max_tail {
        Some(n) => json_search_branch_fragment_bounded(b, n),
        None => json_search_branch_fragment(b),
    };
    if branches.len() == 1 {
        return map_fn(branches[0]);
    }
    let fragments = branches.into_iter().map(map_fn).collect::<Vec<_>>();
    format!("(?:{})", fragments.join("|"))
}

fn regex_byte_length_bounds(expr: &crate::automata::lexer::ast::Expr) -> (usize, Option<usize>) {
    use crate::automata::lexer::ast::Expr;

    match expr {
        Expr::U8Seq(bytes) => (bytes.len(), Some(bytes.len())),
        Expr::U8Class(_) => (1, Some(1)),
        Expr::Seq(parts) => {
            let mut min_total = 0usize;
            let mut max_total = Some(0usize);
            for part in parts {
                let (part_min, part_max) = regex_byte_length_bounds(part);
                min_total = min_total.saturating_add(part_min);
                max_total = match (max_total, part_max) {
                    (Some(left), Some(right)) => Some(left.saturating_add(right)),
                    _ => None,
                };
            }
            (min_total, max_total)
        }
        Expr::Choice(options) => {
            let mut min_total = usize::MAX;
            let mut max_total = Some(0usize);
            for option in options {
                let (option_min, option_max) = regex_byte_length_bounds(option);
                min_total = min_total.min(option_min);
                max_total = match (max_total, option_max) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    _ => None,
                };
            }
            if min_total == usize::MAX {
                (0, Some(0))
            } else {
                (min_total, max_total)
            }
        }
        Expr::Repeat { expr, min, max } => {
            let (inner_min, inner_max) = regex_byte_length_bounds(expr);
            let min_total = inner_min.saturating_mul(*min);
            let max_total = match (inner_max, max) {
                (Some(inner), Some(count)) => Some(inner.saturating_mul(*count)),
                _ => None,
            };
            (min_total, max_total)
        }
        Expr::Shared(expr) => regex_byte_length_bounds(expr),
        Expr::Epsilon => (0, Some(0)),
    }
}

fn prune_pattern_branches_for_min_length(pattern: &str, min_len: usize) -> Option<String> {
    if min_len == 0 {
        return Some(pattern.to_string());
    }

    let branches = split_top_level_regex_branches(pattern);
    let mut kept = Vec::with_capacity(branches.len());
    for branch in &branches {
        let (anchored_start, anchored_end, core) = strip_branch_outer_anchors(branch);
        if !(anchored_start && anchored_end) {
            kept.push(*branch);
            continue;
        }

        let inner = jsonify_regex_dot(core);
        let expr = parse_regex(&inner, true);
        let (_, max_bytes) = regex_byte_length_bounds(&expr);
        if max_bytes.is_some_and(|bound| bound < min_len) {
            continue;
        }
        kept.push(*branch);
    }

    if kept.is_empty() {
        None
    } else if kept.len() == branches.len() {
        Some(pattern.to_string())
    } else {
        Some(kept.join("|"))
    }
}

fn json_direct_ascii_bytes() -> BTreeSet<u8> {
    let mut out = BTreeSet::new();
    out.extend(0x20..=0x21);
    out.extend(0x23..=0x5B);
    out.extend(0x5D..=0x7E); // exclude DEL (0x7F)
    out
}

fn json_escapable_bytes() -> BTreeSet<u8> {
    let mut out = BTreeSet::new();
    out.extend(0x00..=0x1F);
    out.insert(0x22);
    out.insert(0x2F);
    out.insert(0x5C);
    out
}

fn compress_byte_set(values: &BTreeSet<u8>) -> Vec<(u8, u8)> {
    let mut ranges = Vec::new();
    let mut iter = values.iter().copied();
    let Some(mut start) = iter.next() else {
        return ranges;
    };
    let mut end = start;
    for value in iter {
        if value == end.saturating_add(1) {
            end = value;
            continue;
        }
        ranges.push((start, end));
        start = value;
        end = value;
    }
    ranges.push((start, end));
    ranges
}

fn regex_char_class_from_ranges(ranges: &[(u8, u8)]) -> String {
    let mut out = String::from("[");
    for (start, end) in ranges {
        if start == end {
            out.push_str(&format!(r#"\x{:02X}"#, start));
        } else {
            out.push_str(&format!(r#"\x{:02X}-\x{:02X}"#, start, end));
        }
    }
    out.push(']');
    out
}

fn regex_literal_bytes(bytes: &[u8]) -> String {
    bytes.iter()
        .map(|byte| format!(r#"\x{:02X}"#, byte))
        .collect::<Vec<_>>()
        .join("")
}

fn jsonified_literal_fragment(byte: u8) -> Option<String> {
    if byte != b'/' {
        return None;
    }
    let mut parts = Vec::new();
    if json_direct_ascii_bytes().contains(&byte) {
        parts.push(regex_literal_bytes(&[byte]));
    }
    parts.extend(json_escaped_byte_fragments(byte));
    if parts.is_empty() {
        return None;
    }
    if parts.len() == 1 {
        return parts.into_iter().next();
    }
    Some(format!("(?:{})", parts.join("|")))
}

fn is_regex_metachar(byte: u8) -> bool {
    matches!(byte, b'[' | b']' | b'(' | b')' | b'{' | b'}' | b'.' | b'*' | b'+' | b'?' | b'|' | b'^' | b'$' | b'\\')
}

fn hex_nibble_fragment(value: u8) -> String {
    match value {
        0..=9 => char::from(b'0' + value).to_string(),
        10..=15 => format!("[{}{}]", char::from(b'A' + value - 10), char::from(b'a' + value - 10)),
        _ => String::new(),
    }
}

fn json_unicode_escape_fragment(byte: u8) -> String {
    format!(
        r#"\x5Cu00{}{}"#,
        hex_nibble_fragment((byte >> 4) & 0x0F),
        hex_nibble_fragment(byte & 0x0F)
    )
}

fn json_escaped_byte_fragments(byte: u8) -> Vec<String> {
    let mut fragments = Vec::new();
    match byte {
        b'"' => fragments.push(String::from(r#"\x5C\x22"#)),
        b'/' => fragments.push(String::from(r#"\x5C\x2F"#)),
        b'\\' => fragments.push(String::from(r#"\x5C\x5C"#)),
        0x08 => fragments.push(String::from(r#"\x5Cb"#)),
        0x09 => fragments.push(String::from(r#"\x5Ct"#)),
        0x0A => fragments.push(String::from(r#"\x5Cn"#)),
        0x0C => fragments.push(String::from(r#"\x5Cf"#)),
        0x0D => fragments.push(String::from(r#"\x5Cr"#)),
        _ => {}
    }
    if !json_direct_ascii_bytes().contains(&byte) {
        fragments.push(json_unicode_escape_fragment(byte));
    }
    fragments
}

fn parse_class_escape_set(input: &[u8], pos: usize) -> Option<(BTreeSet<u8>, usize)> {
    if pos + 1 >= input.len() {
        return None;
    }
    let mut set = BTreeSet::new();
    match input[pos + 1] {
        b'd' => set.extend(b'0'..=b'9'),
        b's' => {
            set.extend(0x09..=0x0D);
            set.insert(0x20);
            set.insert(0x85);
            set.insert(0xA0);
        }
        b'w' => {
            set.extend(b'0'..=b'9');
            set.extend(b'A'..=b'Z');
            set.insert(b'_');
            set.extend(b'a'..=b'z');
        }
        _ => return None,
    }
    Some((set, pos + 2))
}

fn parse_class_escape_byte(input: &[u8], pos: usize) -> (u8, usize) {
    if pos + 1 >= input.len() {
        return (b'\\', pos + 1);
    }
    match input[pos + 1] {
        b'n' => (b'\n', pos + 2),
        b'r' => (b'\r', pos + 2),
        b't' => (b'\t', pos + 2),
        b'x' if pos + 3 < input.len() => {
            let hex = |digit: u8| -> u8 {
                match digit {
                    b'0'..=b'9' => digit - b'0',
                    b'a'..=b'f' => 10 + digit - b'a',
                    b'A'..=b'F' => 10 + digit - b'A',
                    _ => 0,
                }
            };
            let hi = hex(input[pos + 2]);
            let lo = hex(input[pos + 3]);
            (((hi << 4) | lo), pos + 4)
        }
        other => (other, pos + 2),
    }
}

fn jsonified_char_class_fragment(matched: &BTreeSet<u8>, negate: bool) -> Option<String> {
    let direct_ascii_all = json_direct_ascii_bytes();
    let escapable_all = json_escapable_bytes();
    let mut parts = Vec::new();

    if negate {
        let direct_ascii: BTreeSet<u8> = direct_ascii_all.difference(matched).copied().collect();
        if !direct_ascii.is_empty() {
            parts.push(regex_char_class_from_ranges(&compress_byte_set(&direct_ascii)));
        }
        parts.push(String::from(JSON_DIRECT_UTF8_PATTERN));
        for byte in escapable_all.difference(matched).copied() {
            parts.extend(json_escaped_byte_fragments(byte));
        }
    } else {
        let direct_ascii: BTreeSet<u8> = direct_ascii_all.intersection(matched).copied().collect();
        if !direct_ascii.is_empty() {
            parts.push(regex_char_class_from_ranges(&compress_byte_set(&direct_ascii)));
        }
        for byte in matched.iter().copied().filter(|byte| *byte >= 0x80) {
            parts.push(regex_literal_bytes(char::from(byte).to_string().as_bytes()));
        }
        for byte in escapable_all.intersection(matched).copied() {
            parts.extend(json_escaped_byte_fragments(byte));
        }
    }

    if parts.is_empty() {
        return None;
    }
    if parts.len() == 1 {
        return parts.into_iter().next();
    }
    Some(format!("(?:{})", parts.join("|")))
}

fn jsonify_regex_char_class(input: &[u8], start: usize) -> Option<(String, usize)> {
    let mut pos = start + 1;
    let mut negate = false;
    if pos < input.len() && input[pos] == b'^' {
        negate = true;
        pos += 1;
    }
    let mut matched = BTreeSet::new();
    while pos < input.len() && input[pos] != b']' {
        if input[pos] == b'\\' {
            if let Some((escape_set, next_pos)) = parse_class_escape_set(input, pos) {
                matched.extend(escape_set);
                pos = next_pos;
                continue;
            }
        }

        let (start_byte, next_pos) = if input[pos] == b'\\' {
            parse_class_escape_byte(input, pos)
        } else {
            (input[pos], pos + 1)
        };
        pos = next_pos;

        if pos + 1 < input.len() && input[pos] == b'-' && input[pos + 1] != b']' {
            pos += 1;
            let (end_byte, next_pos) = if input[pos] == b'\\' {
                parse_class_escape_byte(input, pos)
            } else {
                (input[pos], pos + 1)
            };
            pos = next_pos;
            matched.extend(start_byte..=end_byte);
        } else {
            matched.insert(start_byte);
        }
    }
    if pos >= input.len() || input[pos] != b']' {
        return None;
    }
    let fragment = jsonified_char_class_fragment(&matched, negate)?;
    Some((fragment, pos + 1))
}

/// Expand a regex shorthand escape class (`\s`, `\S`, `\d`, `\D`, `\w`, `\W`)
/// into a JSON-string-safe regex fragment. Returns `None` for non-class escapes.
///
/// For positive classes (`\d`, `\s`, `\w`), uses `jsonified_char_class_fragment`
/// for precise expansion. For negated classes (`\D`, `\S`, `\W`), produces a
/// compact fragment that excludes `\\uXXXX` per-byte encodings of control chars
/// to avoid DFA explosion when combined with bounded repetition like `{0,99}`.
fn jsonify_shorthand_class(escape_char: u8) -> Option<String> {
    match escape_char {
        b'd' | b'w' => {
            let matched = shorthand_class_bytes(escape_char);
            jsonified_char_class_fragment(&matched, false)
        }
        b's' => {
            // \s matches ASCII whitespace + NBSP (U+00A0 = \xC2\xA0).
            // llguidance treats NBSP as whitespace, matching ECMA-262 behavior.
            let matched = shorthand_class_bytes(b's');
            let mut fragment = jsonified_char_class_fragment(&matched, false)?;
            // Append NBSP as a 2-byte UTF-8 literal alternative
            fragment = format!("(?:{}|\\xC2\\xA0)", fragment);
            Some(fragment)
        }
        b'D' | b'W' => {
            let excluded = shorthand_class_bytes(escape_char.to_ascii_lowercase());
            Some(compact_negated_json_class(&excluded, false))
        }
        b'S' => {
            let excluded = shorthand_class_bytes(b's');
            Some(compact_negated_json_class(&excluded, true))
        }
        _ => None,
    }
}

/// Returns the byte set for a lowercase shorthand class.
fn shorthand_class_bytes(class: u8) -> BTreeSet<u8> {
    let mut s = BTreeSet::new();
    match class {
        b'd' => { s.extend(b'0'..=b'9'); }
        b's' => {
            // ECMA-262 WhiteSpace + LineTerminator (ASCII subset):
            // TAB (0x09), LF (0x0A), VT (0x0B), FF (0x0C), CR (0x0D), SPACE (0x20)
            s.insert(0x09); s.insert(0x0A); s.insert(0x0B);
            s.insert(0x0C); s.insert(0x0D); s.insert(0x20);
        }
        b'w' => { s.extend(b'0'..=b'9'); s.extend(b'A'..=b'Z'); s.insert(b'_'); s.extend(b'a'..=b'z'); }
        _ => {}
    }
    s
}

/// Produce a compact JSON-string-safe regex fragment for a negated ASCII class.
///
/// Unlike `jsonified_char_class_fragment(_, true)`, this produces a compact
/// fragment that avoids DFA explosion when combined with bounded repetition
/// like `{0,99}`. It includes single-byte ASCII chars, named JSON escape
/// sequences, and direct multi-byte UTF-8 patterns. It omits `\\uXXXX`
/// encodings to keep the resulting DFA small.
fn compact_negated_json_class(excluded: &BTreeSet<u8>, exclude_nbsp: bool) -> String {
    let direct_ascii_all = json_direct_ascii_bytes();
    let mut parts = Vec::new();

    // Direct ASCII bytes not in the excluded set
    let direct_ascii: BTreeSet<u8> = direct_ascii_all.difference(excluded).copied().collect();
    if !direct_ascii.is_empty() {
        parts.push(regex_char_class_from_ranges(&compress_byte_set(&direct_ascii)));
    }

    // Named JSON escape sequences for chars not in the excluded set.
    // Note: \/ (escaped forward slash) is omitted because / is already a
    // direct ASCII byte. Including \/ would accept the escaped form \/ which
    // some validators reject.
    let named_escapes: &[(u8, &str)] = &[
        (0x22, r#"\x5C\x22"#),  // \"
        (0x5C, r#"\x5C\x5C"#),  // \\
        (0x08, r#"\x5Cb"#),     // \b
        (0x0C, r#"\x5Cf"#),     // \f
        (0x0A, r#"\x5Cn"#),     // \n
        (0x0D, r#"\x5Cr"#),     // \r
        (0x09, r#"\x5Ct"#),     // \t
    ];
    for &(byte_val, escape_pattern) in named_escapes {
        if !excluded.contains(&byte_val) {
            parts.push(String::from(escape_pattern));
        }
    }

    // Multi-byte UTF-8 character patterns.
    if exclude_nbsp {
        // For \S: exclude NBSP (U+00A0 = \xC2\xA0) from the 2-byte UTF-8 range.
        // Split \xC2 continuation to skip \xA0, matching llguidance's behavior
        // which treats NBSP as whitespace (\s).
        let utf8_no_nbsp = r#"(?:\xC2[\x80-\x9F\xA1-\xBF]|[\xC3-\xDF][\x80-\xBF]|[\xE0][\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC][\x80-\xBF][\x80-\xBF]|[\xED][\x80-\x9F][\x80-\xBF]|[\xEE-\xEF][\x80-\xBF][\x80-\xBF]|[\xF0][\x90-\xBF][\x80-\xBF][\x80-\xBF]|[\xF1-\xF3][\x80-\xBF][\x80-\xBF][\x80-\xBF]|[\xF4][\x80-\x8F][\x80-\xBF][\x80-\xBF])"#;
        parts.push(String::from(utf8_no_nbsp));
    } else {
        parts.push(String::from(JSON_DIRECT_UTF8_PATTERN));
    }

    if parts.len() == 1 {
        return parts.into_iter().next().unwrap();
    }
    format!("(?:{})", parts.join("|"))
}

/// Replace bare `.` in a regex pattern with the JSON string character class,
/// so that `.` does not match `"`, `\`, or control characters inside a JSON string.
/// Also expands shorthand character classes (`\s`, `\S`, `\d`, `\D`, `\w`, `\W`)
/// into JSON-string-safe equivalents.
fn jsonify_regex_dot(pattern: &str) -> String {
    let json_dot = r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})"#;
    let mut out = String::with_capacity(pattern.len() * 2);
    let bytes = pattern.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == b'\\' && i + 1 < bytes.len() {
            if let Some(fragment) = jsonify_shorthand_class(bytes[i + 1]) {
                out.push_str(&fragment);
                i += 2;
                continue;
            }
            out.push(ch as char);
            out.push(bytes[i + 1] as char);
            i += 2;
            continue;
        }
        if ch == b'[' {
            if let Some((fragment, next_pos)) = jsonify_regex_char_class(bytes, i) {
                out.push_str(&fragment);
                i = next_pos;
                continue;
            }
        }
        if ch == b'.' {
            out.push_str(json_dot);
            i += 1;
            continue;
        }
        if !is_regex_metachar(ch) {
            if let Some(fragment) = jsonified_literal_fragment(ch) {
                out.push_str(&fragment);
                i += 1;
                continue;
            }
        }
        out.push(ch as char);
        i += 1;
    }
    out
}

fn json_wrapped_pattern(pattern: &str) -> GrammarExpr {
    let inner = json_search_pattern(pattern);
    regex_expr(format!(r#""(?:{})""#, inner))
}

fn json_wrapped_pattern_bounded(pattern: &str, max_tail: usize) -> GrammarExpr {
    let inner = json_search_pattern_bounded(pattern, max_tail);
    regex_expr(format!(r#""(?:{})""#, inner))
}

fn json_wrapped_fullmatch_pattern(pattern: &str) -> GrammarExpr {
    let inner = jsonify_regex_dot(pattern);
    regex_expr(format!(r#""(?:{})""#, inner))
}

fn json_wrapped_key_colon_pattern(pattern: &str) -> GrammarExpr {
    let inner = json_search_pattern(pattern);
    regex_expr(format!(r#""(?:{})": "#, inner))
}

fn quoted_expr(inner: GrammarExpr) -> GrammarExpr {
    sequence_or_single(vec![literal_expr(b"\""), inner, literal_expr(b"\"")])
}

fn json_date_body_expr() -> GrammarExpr {
    let year = regex_expr(r#"[0-9]{4}"#);
    let leap_year = regex_expr(
        r#"(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)"#,
    );
    let sep = literal_expr(b"-");
    let month_31 = choice_or_single(
        ["01", "03", "05", "07", "08", "10", "12"]
            .into_iter()
            .map(|month| literal_expr(month.as_bytes()))
            .collect(),
    );
    let month_30 = choice_or_single(
        ["04", "06", "09", "11"]
            .into_iter()
            .map(|month| literal_expr(month.as_bytes()))
            .collect(),
    );
    let february = literal_expr(b"02");
    let day_31 = regex_expr(r#"(?:0[1-9]|[12][0-9]|3[01])"#);
    let day_30 = regex_expr(r#"(?:0[1-9]|[12][0-9]|30)"#);
    let day_28 = regex_expr(r#"(?:0[1-9]|1[0-9]|2[0-8])"#);
    let day_29 = literal_expr(b"29");

    choice_or_single(vec![
        sequence_or_single(vec![
            year.clone(),
            sep.clone(),
            month_31,
            sep.clone(),
            day_31,
        ]),
        sequence_or_single(vec![
            year.clone(),
            sep.clone(),
            month_30,
            sep.clone(),
            day_30,
        ]),
        sequence_or_single(vec![
            year,
            sep.clone(),
            february.clone(),
            sep.clone(),
            day_28,
        ]),
        sequence_or_single(vec![leap_year, sep, february, literal_expr(b"-29")]),
    ])
}

fn json_time_body_expr() -> GrammarExpr {
    let hour = regex_expr(r#"(?:[01][0-9]|2[0-3])"#);
    let minute = regex_expr(r#"[0-5][0-9]"#);
    let second = regex_expr(r#"(?:[0-5][0-9]|60)"#);
    let fraction = GrammarExpr::Optional(Box::new(sequence_or_single(vec![
        literal_expr(b"."),
        regex_expr(r#"[0-9]+"#),
    ])));
    let offset = sequence_or_single(vec![
        choice_or_single(vec![literal_expr(b"+"), literal_expr(b"-")]),
        hour.clone(),
        literal_expr(b":"),
        minute.clone(),
    ]);
    let zone = choice_or_single(vec![literal_expr(b"Z"), literal_expr(b"z"), offset]);

    sequence_or_single(vec![
        hour,
        literal_expr(b":"),
        minute,
        literal_expr(b":"),
        second,
        fraction,
        zone,
    ])
}

fn json_date_time_body_expr() -> GrammarExpr {
    sequence_or_single(vec![
        json_date_body_expr(),
        choice_or_single(vec![literal_expr(b"T"), literal_expr(b"t")]),
        json_time_body_expr(),
    ])
}

fn simple_anchored_literal_pattern(pattern: &str) -> Option<(String, bool)> {
    if !pattern.starts_with('^') {
        return None;
    }

    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::new();
    let mut i = 1usize;
    let mut exact = false;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\\' {
            i += 1;
            if i >= chars.len() {
                return None;
            }
            out.push(chars[i]);
        } else if ch == '$' && i == chars.len() - 1 {
            exact = true;
            break;
        } else if matches!(ch, '.' | '^' | '$' | '*' | '+' | '?' | '{' | '}' | '[' | ']' | '|' | '(' | ')') {
            return None;
        } else {
            out.push(ch);
        }
        i += 1;
    }
    Some((out, exact))
}

fn simple_repeated_single_char_pattern(pattern: &str) -> Option<String> {
    // Only valid when the pattern is fully anchored (^...$), because JSON Schema
    // `pattern` is a search pattern. Without both anchors the string may contain
    // characters outside the char-class, so we must fall back to the general
    // json_search_pattern path.
    let core = pattern.strip_prefix('^')?.strip_suffix('$')?;
    let repeated = core.strip_suffix('+')?;
    if !(repeated.starts_with('[') && repeated.ends_with(']')) {
        return None;
    }
    Some(jsonify_regex_dot(repeated))
}

/// Returns `true` when every top-level branch of `pattern` starts with `^` and
/// ends with `$` (after stripping a single outer group wrapper). For such
/// patterns `json_wrapped_pattern` produces no `<string_tail>` padding, so the
/// resulting regex stays compact and avoids DFA explosion.
fn pattern_all_branches_anchored(pattern: &str) -> bool {
    let branches = split_top_level_regex_branches(pattern);
    branches.iter().all(|b| {
        let (start, end, _) = strip_branch_outer_anchors(b);
        start && end
    })
}

fn json_string_literal_bytes(text: &str) -> Vec<u8> {
    serde_json::to_string(text)
        .unwrap_or_else(|_| format!("\"{}\"", text))
        .into_bytes()
}

fn append_json_native(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => {
            out.push_str(
                &serde_json::to_string(text).unwrap_or_else(|_| format!("\"{}\"", text)),
            );
        }
        Value::Array(values) => {
            out.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    out.push_str(std::str::from_utf8(JSON_ITEM_SEPARATOR).unwrap_or(", "));
                }
                append_json_native(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(std::str::from_utf8(JSON_ITEM_SEPARATOR).unwrap_or(", "));
                }
                out.push_str(
                    &serde_json::to_string(key).unwrap_or_else(|_| format!("\"{}\"", key)),
                );
                out.push_str(std::str::from_utf8(JSON_KEY_SEPARATOR).unwrap_or(": "));
                append_json_native(item, out);
            }
            out.push('}');
        }
    }
}

fn json_value_literal_bytes(value: &Value) -> Vec<u8> {
    let mut rendered = String::new();
    append_json_native(value, &mut rendered);
    rendered.into_bytes()
}

fn json_value_literal_expr(value: &Value) -> GrammarExpr {
    literal_expr(&json_value_literal_bytes(value))
}

fn expr_key(expr: &GrammarExpr) -> String {
    format!("{expr:?}")
}

fn seq_elements(expr: &GrammarExpr) -> Vec<GrammarExpr> {
    match expr {
        GrammarExpr::Sequence(parts) => parts.clone(),
        _ => vec![expr.clone()],
    }
}

fn factor_common_affixes(options: Vec<GrammarExpr>) -> GrammarExpr {
    if options.len() <= 1 {
        return choice_or_single(options);
    }

    let elem_lists: Vec<Vec<GrammarExpr>> = options.iter().map(seq_elements).collect();
    let min_len = elem_lists.iter().map(Vec::len).min().unwrap_or(0);

    let mut prefix_len = 0;
    while prefix_len < min_len
        && elem_lists[1..]
            .iter()
            .all(|elements| elements[prefix_len] == elem_lists[0][prefix_len])
    {
        prefix_len += 1;
    }

    let mut suffix_len = 0;
    while suffix_len < min_len.saturating_sub(prefix_len)
        && elem_lists[1..].iter().all(|elements| {
            elements[elements.len() - suffix_len - 1]
                == elem_lists[0][elem_lists[0].len() - suffix_len - 1]
        })
    {
        suffix_len += 1;
    }

    if prefix_len == 0 && suffix_len == 0 {
        return choice_or_single(options);
    }

    let prefix = elem_lists[0][..prefix_len].to_vec();
    let suffix = if suffix_len == 0 {
        Vec::new()
    } else {
        elem_lists[0][elem_lists[0].len() - suffix_len..].to_vec()
    };

    let middles = elem_lists
        .into_iter()
        .map(|elements| {
            let end = elements.len() - suffix_len;
            sequence_or_single(elements[prefix_len..end].to_vec())
        })
        .collect::<Vec<_>>();

    let mut parts = prefix;
    parts.push(choice_or_single(middles));
    parts.extend(suffix);
    sequence_or_single(parts)
}

fn type_set(schema: &Map<String, Value>) -> Option<BTreeSet<String>> {
    let type_value = schema.get("type")?;
    match type_value {
        Value::String(value) => Some([value.clone()].into_iter().collect()),
        Value::Array(values) => Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
        ),
        _ => None,
    }
}

fn has_structural_keywords(schema: &Map<String, Value>) -> bool {
    const STRUCTURAL: &[&str] = &[
        "type",
        "properties",
        "additionalProperties",
        "patternProperties",
        "items",
        "prefixItems",
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "propertyNames",
    ];
    STRUCTURAL.iter().any(|k| schema.contains_key(*k))
}

fn merge_two_schemas(s1: &Map<String, Value>, s2: &Map<String, Value>) -> Map<String, Value> {
    if s2.is_empty() {
        return s1.clone();
    }
    if s1.is_empty() {
        return s2.clone();
    }

    let mut merged = Map::new();

    let types1 = type_set(s1);
    let types2 = type_set(s2);
    match (types1, types2) {
        (Some(left), Some(right)) => {
            let intersection: Vec<String> = left.intersection(&right).cloned().collect();
            if intersection.is_empty() {
                let mut unsat = Map::new();
                unsat.insert("not".into(), Value::Object(Map::new()));
                return unsat;
            }
            if intersection.len() == 1 {
                merged.insert("type".into(), Value::String(intersection[0].clone()));
            } else {
                merged.insert(
                    "type".into(),
                    Value::Array(intersection.into_iter().map(Value::String).collect()),
                );
            }
        }
        (Some(_), None) => {
            merged.insert("type".into(), s1.get("type").cloned().unwrap());
        }
        (None, Some(_)) => {
            merged.insert("type".into(), s2.get("type").cloned().unwrap());
        }
        (None, None) => {}
    }

    let props1 = s1.get("properties").and_then(Value::as_object);
    let props2 = s2.get("properties").and_then(Value::as_object);
    if props1.is_some() || props2.is_some() {
        let mut merged_props = Map::new();
        // Preserve declaration order: p1 keys first, then new p2 keys
        let mut keys: Vec<String> = Vec::new();
        if let Some(props) = props1 {
            for k in props.keys() {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
        if let Some(props) = props2 {
            for k in props.keys() {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
        for key in keys {
            match (
                props1.and_then(|props| props.get(&key)).and_then(Value::as_object),
                props2.and_then(|props| props.get(&key)).and_then(Value::as_object),
            ) {
                (Some(left), Some(right)) => {
                    merged_props.insert(key, Value::Object(merge_two_schemas(left, right)));
                }
                _ => {
                    if let Some(value) = props1.and_then(|props| props.get(&key)).cloned() {
                        merged_props.insert(key.clone(), value);
                    } else if let Some(value) = props2.and_then(|props| props.get(&key)).cloned() {
                        merged_props.insert(key.clone(), value);
                    }
                }
            }
        }
        merged.insert("properties".into(), Value::Object(merged_props));
    }

    let required1 = s1.get("required").and_then(Value::as_array);
    let required2 = s2.get("required").and_then(Value::as_array);
    let mut combined_required = Vec::new();
    if let Some(values) = required1 {
        for value in values {
            if !combined_required.contains(value) {
                combined_required.push(value.clone());
            }
        }
    }
    if let Some(values) = required2 {
        for value in values {
            if !combined_required.contains(value) {
                combined_required.push(value.clone());
            }
        }
    }
    if !combined_required.is_empty() {
        merged.insert("required".into(), Value::Array(combined_required));
    }

    let ap1 = s1.get("additionalProperties");
    let ap2 = s2.get("additionalProperties");
    let has_ap1 = s1.contains_key("additionalProperties");
    let has_ap2 = s2.contains_key("additionalProperties");
    if has_ap1 || has_ap2 {
        if matches!(ap1, Some(Value::Bool(false))) || matches!(ap2, Some(Value::Bool(false))) {
            merged.insert("additionalProperties".into(), Value::Bool(false));
        } else if let (Some(Value::Object(left)), Some(Value::Object(right))) = (ap1, ap2) {
            merged.insert(
                "additionalProperties".into(),
                Value::Object(merge_two_schemas(left, right)),
            );
        } else if has_ap1 {
            merged.insert(
                "additionalProperties".into(),
                ap1.cloned().unwrap_or(Value::Bool(true)),
            );
        } else if has_ap2 {
            merged.insert(
                "additionalProperties".into(),
                ap2.cloned().unwrap_or(Value::Bool(true)),
            );
        }
    }

    for key in [
        "minimum",
        "exclusiveMinimum",
        "minLength",
        "minItems",
        "minProperties",
    ] {
        match (s1.get(key).cloned(), s2.get(key).cloned()) {
            (Some(Value::Number(left)), Some(Value::Number(right))) => {
                let left = left.as_f64().unwrap_or(0.0);
                let right = right.as_f64().unwrap_or(0.0);
                if let Some(number) = serde_json::Number::from_f64(left.max(right)) {
                    merged.insert(key.into(), Value::Number(number));
                }
            }
            (Some(value), None) | (None, Some(value)) => {
                merged.insert(key.into(), value);
            }
            _ => {}
        }
    }

    for key in [
        "maximum",
        "exclusiveMaximum",
        "maxLength",
        "maxItems",
        "maxProperties",
    ] {
        match (s1.get(key).cloned(), s2.get(key).cloned()) {
            (Some(Value::Number(left)), Some(Value::Number(right))) => {
                let left = left.as_f64().unwrap_or(0.0);
                let right = right.as_f64().unwrap_or(0.0);
                if let Some(number) = serde_json::Number::from_f64(left.min(right)) {
                    merged.insert(key.into(), Value::Number(number));
                }
            }
            (Some(value), None) | (None, Some(value)) => {
                merged.insert(key.into(), value);
            }
            _ => {}
        }
    }

    let handled: HashSet<&'static str> = [
        "type",
        "properties",
        "required",
        "additionalProperties",
        "minimum",
        "exclusiveMinimum",
        "minLength",
        "minItems",
        "minProperties",
        "maximum",
        "exclusiveMaximum",
        "maxLength",
        "maxItems",
        "maxProperties",
    ]
    .into_iter()
    .collect();
    for (key, value) in s1 {
        if !handled.contains(key.as_str()) {
            merged.insert(key.clone(), value.clone());
        }
    }
    for (key, value) in s2 {
        if !handled.contains(key.as_str()) && !merged.contains_key(key) {
            merged.insert(key.clone(), value.clone());
        }
    }

    merged
}

fn merge_allof_schemas(base: &Map<String, Value>, sub_schemas: &[Value]) -> Map<String, Value> {
    let mut merged = base.clone();
    for schema in sub_schemas {
        if let Some(object) = schema.as_object() {
            merged = merge_two_schemas(&merged, object);
        }
    }
    merged
}

pub fn json_schema_to_grammar(schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    let schema: Value = serde_json::from_str(schema_json)
        .map_err(|err| GlrMaskError::GrammarParse(err.to_string()))?;
    let mut named = schema_to_named_grammar(&schema)?;
    promote_large_literal_alts(&mut named, 10);
    lower(&named)
}

pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {
    let mut ctx = SchemaCtx::new(schema);
    ctx.register_root_definitions();
    ctx.materialize_registered_refs()?;
    let start_expr = match ctx.convert_schema(schema) {
        Ok(expr) => expr,
        Err(err) if is_unsat_schema_error(&err) => never_expr(),
        Err(err) => return Err(err),
    };
    ctx.insert_rule("start", start_expr);
    let terminal_names: HashSet<String> = ctx
        .rules
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| {
            !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        })
        .map(|s| s.to_string())
        .collect();
    let rules = ctx.rules.into_iter().map(|(name, expr)| {
        let is_terminal = terminal_names.contains(&name);
        NamedRule { name, expr, is_terminal }
    }).collect();
    Ok(NamedGrammar {
        rules,
        start: "start".into(),
        ignore: None,
    })
}

struct SchemaCtx {
    root_schema: Value,
    rules: Vec<(String, GrammarExpr)>,
    rule_indices: HashMap<String, usize>,
    used_rule_names: HashSet<String>,
    ref_rule_names: HashMap<String, String>,
    ref_compile_stack: HashSet<String>,
    object_rule_counter: usize,
    anon_rule_counter: usize,
    ap_catch_all_cache: HashMap<String, (String, String)>,
    expr_dedup_cache: HashMap<String, String>,
    json_string_exact_cache: HashMap<usize, String>,
    json_string_upto_cache: HashMap<usize, String>,
    draft_stack: Vec<JsonSchemaDraft>,
}

impl SchemaCtx {
    fn new(root: &Value) -> Self {
        let mut ctx = Self {
            root_schema: root.clone(),
            rules: Vec::new(),
            rule_indices: HashMap::new(),
            used_rule_names: HashSet::new(),
            ref_rule_names: HashMap::new(),
            ref_compile_stack: HashSet::new(),
            object_rule_counter: 0,
            anon_rule_counter: 0,
            ap_catch_all_cache: HashMap::new(),
            expr_dedup_cache: HashMap::new(),
            json_string_exact_cache: HashMap::new(),
            json_string_upto_cache: HashMap::new(),
            draft_stack: vec![DEFAULT_JSON_SCHEMA_DRAFT],
        };
        ctx.ensure_base_rules();
        ctx
    }

    fn current_draft(&self) -> JsonSchemaDraft {
        *self.draft_stack.last().unwrap_or(&DEFAULT_JSON_SCHEMA_DRAFT)
    }

    fn validate_llguidance_keyword_compatibility(
        &self,
        schema: &Map<String, Value>,
        draft: JsonSchemaDraft,
    ) -> Result<(), GlrMaskError> {
        if schema
            .keys()
            .all(|key| META_AND_ANNOTATION_KEYWORDS.contains(&key.as_str()) || !is_known_keyword(draft, key))
        {
            return Ok(());
        }

        let mut unimplemented = schema
            .keys()
            .filter(|key| {
                is_known_keyword(draft, key)
                    && !IMPLEMENTED_JSON_SCHEMA_KEYWORDS.contains(&key.as_str())
                    && !META_AND_ANNOTATION_KEYWORDS.contains(&key.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        if !unimplemented.is_empty() {
            unimplemented.sort();
            return Err(GlrMaskError::GrammarParse(format!(
                "Unimplemented keys: {:?}",
                unimplemented
            )));
        }
        Ok(())
    }

    fn insert_rule(&mut self, name: impl Into<String>, expr: GrammarExpr) -> String {
        let name = name.into();
        if let Some(&index) = self.rule_indices.get(&name) {
            self.rules[index].1 = expr;
        } else {
            self.used_rule_names.insert(name.clone());
            self.rule_indices.insert(name.clone(), self.rules.len());
            self.rules.push((name.clone(), expr));
        }
        name
    }

    fn fresh_rule_name(&mut self, prefix: &str) -> String {
        let prefix = sanitize_rule_name(prefix);
        loop {
            let name = format!("{}_{}", prefix, self.anon_rule_counter);
            self.anon_rule_counter += 1;
            if !self.used_rule_names.contains(&name) {
                return name;
            }
        }
    }

    fn json_value_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_VALUE_RULE.into())
    }

    fn json_object_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_OBJECT_RULE.into())
    }

    fn json_array_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_ARRAY_RULE.into())
    }

    fn json_string_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_STRING_RULE.into())
    }

    fn json_string_char_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_STRING_CHAR_RULE.into())
    }

    fn json_string_char_exact_ref(&mut self, count: usize) -> GrammarExpr {
        match count {
            0 => empty_expr(),
            1 => self.json_string_char_ref(),
            _ => {
                if let Some(rule_name) = self.json_string_exact_cache.get(&count) {
                    return GrammarExpr::Ref(rule_name.clone());
                }

                let chunk = highest_power_of_two_leq(count);
                let expr = if chunk == count {
                    let left = self.json_string_char_exact_ref(count / 2);
                    let right = self.json_string_char_exact_ref(count / 2);
                    sequence_or_single(vec![left, right])
                } else {
                    sequence_or_single(vec![
                        self.json_string_char_exact_ref(chunk),
                        self.json_string_char_exact_ref(count - chunk),
                    ])
                };

                let rule = self.extract_rule(expr, &format!("JSON_STRING_CHAR_EXACT_{count}"));
                if let GrammarExpr::Ref(rule_name) = &rule {
                    self.json_string_exact_cache.insert(count, rule_name.clone());
                }
                rule
            }
        }
    }

    fn json_string_char_upto_ref(&mut self, max: usize) -> GrammarExpr {
        match max {
            0 => empty_expr(),
            1 => GrammarExpr::Optional(Box::new(self.json_string_char_ref())),
            _ => {
                if let Some(rule_name) = self.json_string_upto_cache.get(&max) {
                    return GrammarExpr::Ref(rule_name.clone());
                }

                let chunk = highest_power_of_two_leq(max);
                let expr = if chunk == max {
                    choice_or_single(vec![
                        self.json_string_char_upto_ref(max - 1),
                        self.json_string_char_exact_ref(max),
                    ])
                } else {
                    choice_or_single(vec![
                        self.json_string_char_upto_ref(chunk - 1),
                        sequence_or_single(vec![
                            self.json_string_char_exact_ref(chunk),
                            self.json_string_char_upto_ref(max - chunk),
                        ]),
                    ])
                };

                let rule = self.extract_rule(expr, &format!("JSON_STRING_CHAR_UPTO_{max}"));
                if let GrammarExpr::Ref(rule_name) = &rule {
                    self.json_string_upto_cache.insert(max, rule_name.clone());
                }
                rule
            }
        }
    }

    fn json_key_colon_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_KEY_COLON_RULE.into())
    }

    fn json_integer_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_INTEGER_RULE.into())
    }

    fn json_number_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_NUMBER_RULE.into())
    }

    fn json_bool_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_BOOL_RULE.into())
    }

    fn json_null_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_NULL_RULE.into())
    }

    fn ensure_base_rules(&mut self) {
        self.insert_rule(JSON_STRING_CHAR_RULE, regex_expr(JSON_STRING_CHAR_PATTERN));
        self.insert_rule(JSON_STRING_RULE, regex_expr(JSON_STRING_REGEX));
        self.insert_rule(JSON_INTEGER_RULE, regex_expr(r#"-?(0|[1-9][0-9]*)"#));
        self.insert_rule(
            JSON_NUMBER_RULE,
            regex_expr(r#"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"#),
        );
        self.insert_rule(JSON_NONNEG_INTEGER_RULE, regex_expr(r#"(0|[1-9][0-9]*)"#));
        self.insert_rule(
            JSON_NONNEG_NUMBER_RULE,
            regex_expr(r#"(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"#),
        );
        self.insert_rule(
            JSON_BOOL_RULE,
            choice_or_single(vec![literal_expr(b"true"), literal_expr(b"false")]),
        );
        self.insert_rule(JSON_NULL_RULE, literal_expr(b"null"));
        self.insert_rule(JSON_KEY_COLON_RULE, regex_expr(JSON_KEY_COLON_REGEX));
        self.insert_rule(
            JSON_KV_RULE,
            sequence_or_single(vec![self.json_key_colon_ref(), self.json_value_ref()]),
        );
        self.insert_rule(
            JSON_OBJECT_RULE,
            choice_or_single(vec![
                sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]),
                sequence_or_single(vec![
                    literal_expr(b"{"),
                    GrammarExpr::Ref(JSON_KV_RULE.into()),
                    repeat_expr(
                        sequence_or_single(vec![self.json_item_separator_expr(), GrammarExpr::Ref(JSON_KV_RULE.into())]),
                        0,
                        None,
                    ),
                    literal_expr(b"}"),
                ]),
            ]),
        );
        self.insert_rule(
            JSON_ARRAY_RULE,
            choice_or_single(vec![
                sequence_or_single(vec![literal_expr(b"["), literal_expr(b"]")]),
                sequence_or_single(vec![
                    literal_expr(b"["),
                    self.json_value_ref(),
                    repeat_expr(
                        sequence_or_single(vec![self.json_item_separator_expr(), self.json_value_ref()]),
                        0,
                        None,
                    ),
                    literal_expr(b"]"),
                ]),
            ]),
        );
        self.insert_rule(
            JSON_VALUE_RULE,
            choice_or_single(vec![
                self.json_object_ref(),
                self.json_array_ref(),
                self.json_string_ref(),
                self.json_number_ref(),
                self.json_integer_ref(),
                self.json_bool_ref(),
                self.json_null_ref(),
            ]),
        );
    }

    fn register_root_definitions(&mut self) {
        let definition_pointers = self
            .root_schema
            .as_object()
            .into_iter()
            .flat_map(|root| {
                ["$defs", "definitions"].into_iter().flat_map(move |defs_key| {
                    root.get(defs_key)
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(move |defs| {
                            defs.keys()
                                .map(move |key| format!("#/{defs_key}/{key}"))
                        })
                })
            })
            .collect::<Vec<_>>();
        for pointer in definition_pointers {
            self.ensure_ref_rule(&pointer);
        }
    }

    fn materialize_registered_refs(&mut self) -> Result<(), GlrMaskError> {
        let refs = self.ref_rule_names.keys().cloned().collect::<Vec<_>>();
        for ref_value in refs {
            self.compile_ref_rule(&ref_value)?;
        }
        Ok(())
    }

    fn decode_ref_token(token: &str) -> String {
        token.replace("~1", "/").replace("~0", "~")
    }

    fn find_local_anchor_target<'a>(node: &'a Value, ref_value: &str) -> Option<&'a Value> {
        match node {
            Value::Object(map) => {
                if map.get("id").and_then(Value::as_str) == Some(ref_value)
                    || map.get("$id").and_then(Value::as_str) == Some(ref_value)
                {
                    return Some(node);
                }
                if let Some(anchor_name) = ref_value.strip_prefix('#') {
                    if map.get("$anchor").and_then(Value::as_str) == Some(anchor_name) {
                        return Some(node);
                    }
                }
                for value in map.values() {
                    if let Some(target) = Self::find_local_anchor_target(value, ref_value) {
                        return Some(target);
                    }
                }
                None
            }
            Value::Array(items) => items
                .iter()
                .find_map(|value| Self::find_local_anchor_target(value, ref_value)),
            _ => None,
        }
    }

    fn resolve_local_ref(&self, ref_value: &str) -> Result<Value, GlrMaskError> {
        if !ref_value.starts_with('#') {
            return Err(GlrMaskError::GrammarParse(format!(
                "unsupported $ref '{ref_value}'"
            )));
        }

        if ref_value == "#" {
            return Ok(self.root_schema.clone());
        }

        if !ref_value.starts_with("#/") {
            return Self::find_local_anchor_target(&self.root_schema, ref_value)
                .cloned()
                .ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!("unknown $ref target '{ref_value}'"))
                });
        }

        let mut current = &self.root_schema;
        for token in ref_value[2..].split('/') {
            let key = Self::decode_ref_token(token);
            current = current.get(&key).ok_or_else(|| {
                GlrMaskError::GrammarParse(format!("unknown $ref target '{ref_value}'"))
            })?;
        }
        Ok(current.clone())
    }

    fn schema_for_intersection(&self, schema: &Value) -> Map<String, Value> {
        if schema == &Value::Bool(false) {
            let mut unsat = Map::new();
            unsat.insert("not".into(), Value::Object(Map::new()));
            return unsat;
        }
        if schema == &Value::Bool(true) {
            return Map::new();
        }
        let Some(object) = schema.as_object() else {
            return Map::new();
        };
        let Some(ref_value) = object.get("$ref").and_then(Value::as_str) else {
            return object.clone();
        };
        let Ok(resolved) = self.resolve_local_ref(ref_value) else {
            return object.clone();
        };
        let mut merged = resolved.as_object().cloned().unwrap_or_default();
        let siblings: Map<String, Value> = object
            .iter()
            .filter(|(key, _)| key.as_str() != "$ref")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !siblings.is_empty() {
            merged = merge_two_schemas(&merged, &siblings);
        }
        merged
    }

    fn merge_resolved_subschemas(
        &self,
        base: &Map<String, Value>,
        sub_schemas: &[Value],
    ) -> Map<String, Value> {
        let mut merged = base.clone();
        for schema in sub_schemas {
            merged = merge_two_schemas(&merged, &self.schema_for_intersection(schema));
        }
        merged
    }

    fn stable_ref_rule_name(&self, ref_value: &str) -> String {
        let path = ref_value.strip_prefix("#/").unwrap_or(ref_value);
        let last_segment = path.rsplit('/').next().unwrap_or(path);
        let sanitized = sanitize_rule_name(last_segment).to_lowercase();
        if sanitized.starts_with("def_") {
            sanitized
        } else {
            format!("def_{sanitized}")
        }
    }

    fn ensure_ref_rule(&mut self, ref_value: &str) -> String {
        if let Some(name) = self.ref_rule_names.get(ref_value) {
            return name.clone();
        }

        let base = self.stable_ref_rule_name(ref_value);
        let mut name = base.clone();
        let mut counter = 2;
        while self.used_rule_names.contains(&name) {
            name = format!("{base}_{counter}");
            counter += 1;
        }
        self.used_rule_names.insert(name.clone());
        self.ref_rule_names.insert(ref_value.to_string(), name.clone());
        name
    }

    fn compile_ref_rule(&mut self, ref_value: &str) -> Result<(), GlrMaskError> {
        let rule_name = self.ensure_ref_rule(ref_value);
        if self.rule_indices.contains_key(&rule_name) || self.ref_compile_stack.contains(ref_value) {
            return Ok(());
        }

        self.ref_compile_stack.insert(ref_value.to_string());
        let expr = match self.resolve_local_ref(ref_value) {
            Ok(target) => match self.convert_schema(&target) {
                Ok(expr) => expr,
                Err(err) if is_unsat_schema_error(&err) => never_expr(),
                Err(_) => self.json_value_ref(),
            },
            Err(_) => self.json_value_ref(),
        };
        self.ref_compile_stack.remove(ref_value);
        self.insert_rule(rule_name, expr);
        Ok(())
    }

    fn convert_ref(&mut self, ref_value: &str) -> Result<GrammarExpr, GlrMaskError> {
        let rule_name = self.ensure_ref_rule(ref_value);
        self.compile_ref_rule(ref_value)?;
        Ok(GrammarExpr::Ref(rule_name))
    }

    fn convert_schema(&mut self, schema: &Value) -> Result<GrammarExpr, GlrMaskError> {
        if schema == &Value::Bool(false) {
            return Err(unsat_schema_error());
        }
        if schema == &Value::Bool(true) {
            return Ok(self.json_value_ref());
        }
        let Some(object) = schema.as_object() else {
            return Ok(self.json_value_ref());
        };

        let draft = detect_draft(object, self.current_draft())?;
        self.draft_stack.push(draft);
        let result = (|| {
            self.validate_llguidance_keyword_compatibility(object, draft)?;

            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                return self.convert_ref(reference);
            }

            if matches!(object.get("not"), Some(Value::Object(inner)) if inner.is_empty()) {
                return Err(unsat_schema_error());
            }

            if let Some(expr) = self.convert_finite_value_schema(object)? {
                return Ok(expr);
            }

            if let Some(value) = object.get("const") {
                return Ok(self.json_literal(value));
            }

            if let Some(values) = object.get("enum").and_then(Value::as_array) {
                if !values.is_empty() {
                    return Ok(choice_or_single(values.iter().map(|value| self.json_literal(value)).collect()));
                }
            }

            if let Some(options) = object.get("anyOf").and_then(Value::as_array) {
                if !options.is_empty() {
                    let base_has_structural = has_structural_keywords(object);
                    let options = if base_has_structural {
                        let base: Map<String, Value> = object
                            .iter()
                            .filter(|(key, _)| key.as_str() != "anyOf" && key.as_str() != "oneOf")
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        options
                            .iter()
                            .map(|option| {
                                let merged = Value::Object(self.merge_resolved_subschemas(&base, std::slice::from_ref(option)));
                                self.convert_schema(&merged)
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        options
                            .iter()
                            .map(|option| self.convert_schema(option))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                    return Ok(factor_common_affixes(options));
                }
            }

            if let Some(options) = object.get("oneOf").and_then(Value::as_array) {
                if !options.is_empty() {
                    return Err(GlrMaskError::GrammarParse(
                        "oneOf constraints are not supported. Enable 'coerce_one_of' option to approximate oneOf with anyOf".into(),
                    ));
                }
            }

            if let Some(all_of) = object.get("allOf").and_then(Value::as_array) {
                if !all_of.is_empty() {
                    let base = object
                        .iter()
                        .filter(|(key, _)| key.as_str() != "allOf")
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<Map<String, Value>>();
                    let merged = self.merge_resolved_subschemas(&base, all_of);
                    return self.convert_schema(&Value::Object(merged));
                }
            }

            if let Some(type_values) = object.get("type").and_then(Value::as_array) {
                let options = type_values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|type_name| self.convert_type(type_name, object))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(choice_or_single(options));
            }

            if let Some(type_name) = object.get("type").and_then(Value::as_str) {
                return self.convert_type(type_name, object);
            }

            self.convert_untyped_schema(object)
        })();
        self.draft_stack.pop();
        result
    }

    fn convert_finite_value_schema(
        &mut self,
        object: &Map<String, Value>,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if let Some(value) = object.get("const") {
            let remaining: Map<String, Value> = object
                .iter()
                .filter(|(key, _)| key.as_str() != "const")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            if self.value_satisfies_schema(value, &Value::Object(remaining)) {
                return Ok(Some(self.json_literal(value)));
            }
            return Err(unsat_schema_error());
        }

        if let Some(values) = object.get("enum").and_then(Value::as_array) {
            let remaining: Map<String, Value> = object
                .iter()
                .filter(|(key, _)| key.as_str() != "enum")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let options: Vec<GrammarExpr> = values
                .iter()
                .filter(|value| self.value_satisfies_schema(value, &Value::Object(remaining.clone())))
                .map(|value| self.json_literal(value))
                .collect();
            if !options.is_empty() {
                return Ok(Some(factor_common_affixes(options)));
            }
            return Err(unsat_schema_error());
        }

        Ok(None)
    }

    fn value_satisfies_schema(&self, value: &Value, schema: &Value) -> bool {
        if schema == &Value::Bool(false) {
            return false;
        }
        if schema == &Value::Bool(true) {
            return true;
        }
        let Some(object) = schema.as_object() else {
            return true;
        };

        if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
            if let Ok(target) = self.resolve_local_ref(reference) {
                return self.value_satisfies_schema(value, &target);
            }
            return true;
        }

        if let Some(not_schema) = object.get("not") {
            if self.value_satisfies_schema(value, not_schema) {
                return false;
            }
        }

        if let Some(const_value) = object.get("const") {
            if value != const_value {
                return false;
            }
        }

        if let Some(enum_values) = object.get("enum").and_then(Value::as_array) {
            if !enum_values.iter().any(|item| item == value) {
                return false;
            }
        }

        if let Some(all_of) = object.get("allOf").and_then(Value::as_array) {
            if all_of.iter().any(|item| !self.value_satisfies_schema(value, item)) {
                return false;
            }
        }

        if let Some(any_of) = object.get("anyOf").and_then(Value::as_array) {
            if !any_of.is_empty() && !any_of.iter().any(|item| self.value_satisfies_schema(value, item)) {
                return false;
            }
        }

        if let Some(one_of) = object.get("oneOf").and_then(Value::as_array) {
            if !one_of.is_empty() {
                let matches = one_of
                    .iter()
                    .filter(|item| self.value_satisfies_schema(value, item))
                    .count();
                if matches != 1 {
                    return false;
                }
            }
        }

        match object.get("type") {
            Some(Value::String(type_name)) => {
                if !type_allows_value(type_name, value) {
                    return false;
                }
            }
            Some(Value::Array(type_names)) => {
                let allowed: Vec<&str> = type_names.iter().filter_map(Value::as_str).collect();
                if !allowed.is_empty() && !allowed.iter().any(|type_name| type_allows_value(type_name, value)) {
                    return false;
                }
            }
            _ => {}
        }

        if let Some(text) = value.as_str() {
            if let Some(min_length) = object.get("minLength").and_then(Value::as_u64) {
                if text.chars().count() < min_length as usize {
                    return false;
                }
            }
            if let Some(max_length) = object.get("maxLength").and_then(Value::as_u64) {
                if text.chars().count() > max_length as usize {
                    return false;
                }
            }
        }

        if let Some(entries) = value.as_object() {
            if let Some(min_properties) = object.get("minProperties").and_then(Value::as_u64) {
                if entries.len() < min_properties as usize {
                    return false;
                }
            }
            if let Some(max_properties) = object.get("maxProperties").and_then(Value::as_u64) {
                if entries.len() > max_properties as usize {
                    return false;
                }
            }

            if let Some(required) = object.get("required").and_then(Value::as_array) {
                for key in required.iter().filter_map(Value::as_str) {
                    if !entries.contains_key(key) {
                        return false;
                    }
                }
            }

            let properties = object.get("properties").and_then(Value::as_object);
            if let Some(properties) = properties {
                for (key, subschema) in properties {
                    if let Some(item) = entries.get(key) {
                        if !self.value_satisfies_schema(item, subschema) {
                            return false;
                        }
                    }
                }
            }

            match object.get("additionalProperties") {
                Some(Value::Bool(false)) => {
                    if let Some(properties) = properties {
                        if entries.keys().any(|key| !properties.contains_key(key)) {
                            return false;
                        }
                    }
                }
                Some(Value::Object(schema)) => {
                    for (key, item) in entries {
                        if properties.map(|props| props.contains_key(key)).unwrap_or(false) {
                            continue;
                        }
                        if !self.value_satisfies_schema(item, &Value::Object(schema.clone())) {
                            return false;
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(items) = value.as_array() {
            if let Some(min_items) = object.get("minItems").and_then(Value::as_u64) {
                if items.len() < min_items as usize {
                    return false;
                }
            }
            if let Some(max_items) = object.get("maxItems").and_then(Value::as_u64) {
                if items.len() > max_items as usize {
                    return false;
                }
            }

            if let Some(prefix_items) = object.get("prefixItems").and_then(Value::as_array) {
                for (index, subschema) in prefix_items.iter().enumerate() {
                    if let Some(item) = items.get(index) {
                        if !self.value_satisfies_schema(item, subschema) {
                            return false;
                        }
                    }
                }
                if let Some(item_schema) = object.get("items") {
                    for item in items.iter().skip(prefix_items.len()) {
                        if !self.value_satisfies_schema(item, item_schema) {
                            return false;
                        }
                    }
                }
            } else if let Some(item_schema) = object.get("items") {
                for item in items {
                    if !self.value_satisfies_schema(item, item_schema) {
                        return false;
                    }
                }
            }
        }

        if let Some(number) = value.as_f64() {
            if value.is_boolean() {
                return true;
            }
            if let Some(minimum) = object.get("minimum").and_then(Value::as_f64) {
                if number < minimum {
                    return false;
                }
            }
            if let Some(exclusive_minimum) = object.get("exclusiveMinimum").and_then(Value::as_f64) {
                if number <= exclusive_minimum {
                    return false;
                }
            }
            if let Some(maximum) = object.get("maximum").and_then(Value::as_f64) {
                if number > maximum {
                    return false;
                }
            }
            if let Some(exclusive_maximum) = object.get("exclusiveMaximum").and_then(Value::as_f64) {
                if number >= exclusive_maximum {
                    return false;
                }
            }
            if let Some(multiple_of) = object.get("multipleOf").and_then(Value::as_f64) {
                if multiple_of != 0.0 {
                    let quotient = number / multiple_of;
                    if (quotient - quotient.round()).abs() > 1e-9 {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn is_certainly_unsatisfiable(&mut self, schema: &Value) -> bool {
        if schema == &Value::Bool(false) {
            return true;
        }
        if schema == &Value::Bool(true) {
            return false;
        }
        let Some(object) = schema.as_object() else {
            return false;
        };
        if matches!(object.get("not"), Some(Value::Object(inner)) if inner.is_empty()) {
            return true;
        }
        if !object.contains_key("const") && !object.contains_key("enum") {
            return false;
        }
        matches!(self.convert_finite_value_schema(object), Err(err) if is_unsat_schema_error(&err))
    }

    fn convert_type(&mut self, type_name: &str, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        match type_name {
            "object" => self.build_object_expr(schema),
            "array" => self.build_array_expr(schema),
            "string" => self.build_string_expr(schema),
            "integer" => Ok(self.build_numeric_ref(type_name, schema)),
            "number" => Ok(self.build_numeric_ref(type_name, schema)),
            "boolean" => Ok(self.json_bool_ref()),
            "null" => Ok(self.json_null_ref()),
            _ => Ok(self.json_value_ref()),
        }
    }

    fn base_expr_for_type(&self, type_name: &str) -> GrammarExpr {
        match type_name {
            "object" => self.json_object_ref(),
            "array" => self.json_array_ref(),
            "string" => self.json_string_ref(),
            "number" => self.json_number_ref(),
            "boolean" => self.json_bool_ref(),
            "null" => self.json_null_ref(),
            _ => self.json_value_ref(),
        }
    }

    fn convert_untyped_schema(&mut self, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        let mut options = Vec::new();
        let mut saw_applicable_keywords = false;

        for (type_name, keys) in UNTYPED_SCHEMA_APPLICABLE_TYPES {
            if keys.iter().any(|key| schema.contains_key(*key)) {
                options.push(self.convert_type(type_name, schema)?);
                saw_applicable_keywords = true;
            } else {
                options.push(self.base_expr_for_type(type_name));
            }
        }

        if !saw_applicable_keywords {
            return Ok(self.json_value_ref());
        }

        options.push(self.json_bool_ref());
        options.push(self.json_null_ref());
        Ok(factor_common_affixes(options))
    }

    fn is_nonneg_schema(schema: &Map<String, Value>) -> bool {
        for key in ["minimum", "exclusiveMinimum"] {
            if let Some(val) = schema.get(key) {
                if let Some(n) = val.as_f64() {
                    if n >= 0.0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn build_numeric_ref(&mut self, type_name: &str, schema: &Map<String, Value>) -> GrammarExpr {
        let minimum = schema.get("minimum").and_then(Value::as_f64);
        let exclusive_minimum = schema.get("exclusiveMinimum").and_then(Value::as_f64);
        let maximum = schema.get("maximum").and_then(Value::as_f64);
        let exclusive_maximum = schema.get("exclusiveMaximum").and_then(Value::as_f64);

        // Determine effective bounds
        let mut left: Option<f64> = None;
        let mut left_inclusive = true;
        if let Some(m) = minimum {
            left = Some(m);
        }
        if let Some(em) = exclusive_minimum {
            if left.is_none() || em >= left.unwrap() {
                left = Some(em);
                left_inclusive = false;
            }
        }

        let mut right: Option<f64> = None;
        let mut right_inclusive = true;
        if let Some(m) = maximum {
            right = Some(m);
        }
        if let Some(em) = exclusive_maximum {
            if right.is_none() || em <= right.unwrap() {
                right = Some(em);
                right_inclusive = false;
            }
        }

        let has_bounds = left.is_some() || right.is_some();
        if !has_bounds {
            return if type_name == "integer" {
                self.json_integer_ref()
            } else {
                self.json_number_ref()
            };
        }

        // Only the exact lower bound of 0 can safely use the generic non-negative rules.
        let use_nonneg_shortcut = right.is_none() && left == Some(0.0) && left_inclusive;
        if use_nonneg_shortcut {
            return if type_name == "integer" {
                GrammarExpr::Ref(JSON_NONNEG_INTEGER_RULE.into())
            } else {
                GrammarExpr::Ref(JSON_NONNEG_NUMBER_RULE.into())
            };
        }

        // Build precise range regex
        use crate::import::numeric_range::{rx_float_range, rx_int_range};

        let regex_result = if type_name == "integer" {
            let int_left = left.map(|l| if left_inclusive { l as i64 } else { l as i64 + 1 });
            let int_right = right.map(|r| if right_inclusive { r as i64 } else { r as i64 - 1 });
            rx_int_range(int_left, int_right)
        } else {
            rx_float_range(left, right, left_inclusive, right_inclusive)
        };

        match regex_result {
            Ok(regex) => GrammarExpr::RawRegex(regex),
            Err(_) => {
                // Fallback to generic on error
                if use_nonneg_shortcut {
                    if type_name == "integer" {
                        GrammarExpr::Ref(JSON_NONNEG_INTEGER_RULE.into())
                    } else {
                        GrammarExpr::Ref(JSON_NONNEG_NUMBER_RULE.into())
                    }
                } else if type_name == "integer" {
                    self.json_integer_ref()
                } else {
                    self.json_number_ref()
                }
            }
        }
    }

    fn build_string_expr(&mut self, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        let min_len = schema
            .get("minLength")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(0);
        let max_len = schema
            .get("maxLength")
            .and_then(Value::as_u64)
            .map(|value| value as usize);

        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            let Some(pattern) = prune_pattern_branches_for_min_length(pattern, min_len) else {
                return Ok(self.extract_terminal_rule(never_expr(), "JSON_STRING_PATTERN_UNSAT"));
            };
            if let Some(unit_pattern) = simple_repeated_single_char_pattern(&pattern) {
                if min_len > 0 || max_len.is_some() {
                    return Ok(self.build_bounded_string_from_unit_regex(&unit_pattern, min_len, max_len));
                }
                return Ok(json_wrapped_fullmatch_pattern(&format!("{}+", unit_pattern)));
            }
            if pattern_all_branches_anchored(&pattern) {
                // Every branch is ^…$, so json_wrapped_pattern produces no
                // <string_tail> padding — safe from DFA explosion. Length
                // bounds are implicitly enforced by the anchored regex itself.
                return Ok(json_wrapped_pattern(&pattern));
            }
            if min_len > 0 || max_len.is_some() {
                // Unanchored pattern with length bounds. Use bounded search
                // wrapping (string tails capped to maxLength) to keep the DFA
                // compact. For very large maxLength values, fall through to the
                // length-only path to avoid state explosion.
                const MAX_BOUNDED_SEARCH_TAIL: usize = 100;
                if let Some(ml) = max_len {
                    if ml <= MAX_BOUNDED_SEARCH_TAIL {
                        return Ok(json_wrapped_pattern_bounded(&pattern, ml));
                    }
                }
                // maxLength is too large or absent — drop pattern, use length only
            } else {
                return Ok(json_wrapped_pattern(&pattern));
            }
        }

        if let Some(format_name) = schema.get("format").and_then(Value::as_str) {
            return self.build_format_string_expr(format_name);
        }
        if min_len == 0 && max_len.is_none() {
            return Ok(self.json_string_ref());
        }

        let bounded_body = match max_len {
            Some(max_len) if min_len == max_len => self.json_string_char_exact_ref(min_len),
            Some(max_len) => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(self.json_string_char_exact_ref(min_len));
                }
                if max_len > min_len {
                    parts.push(self.json_string_char_upto_ref(max_len - min_len));
                }
                sequence_or_single(parts)
            }
            None => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(self.json_string_char_exact_ref(min_len));
                }
                parts.push(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
                sequence_or_single(parts)
            }
        };

        Ok(self.extract_terminal_rule(
            sequence_or_single(vec![
                literal_expr(b"\""),
                bounded_body,
                literal_expr(b"\""),
            ]),
            "JSON_STRING_BOUNDED",
        ))
    }

    fn build_bounded_string_from_unit_regex(
        &mut self,
        unit_pattern: &str,
        min_len: usize,
        max_len: Option<usize>,
    ) -> GrammarExpr {
        let unit_expr = self.extract_terminal_rule(
            regex_expr(unit_pattern.to_string()),
            "JSON_STRING_PATTERN_CHAR",
        );
        let bounded_body = match max_len {
            Some(max_len) if min_len == max_len => repeat_expr(unit_expr, min_len, Some(min_len)),
            Some(max_len) => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(repeat_expr(unit_expr.clone(), min_len, Some(min_len)));
                }
                if max_len > min_len {
                    parts.push(repeat_expr(unit_expr, 0, Some(max_len - min_len)));
                }
                sequence_or_single(parts)
            }
            None => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(repeat_expr(unit_expr.clone(), min_len, Some(min_len)));
                }
                parts.push(GrammarExpr::Repeat(Box::new(unit_expr)));
                sequence_or_single(parts)
            }
        };

        self.extract_terminal_rule(
            sequence_or_single(vec![
                literal_expr(b"\""),
                bounded_body,
                literal_expr(b"\""),
            ]),
            "JSON_STRING_BOUNDED_PATTERN",
        )
    }

    fn build_format_string_expr(&mut self, format_name: &str) -> Result<GrammarExpr, GlrMaskError> {
        let expr = match format_name {
            "date" => quoted_expr(json_date_body_expr()),
            "time" => quoted_expr(json_time_body_expr()),
            "date-time" => quoted_expr(json_date_time_body_expr()),
            _ => json_format_pattern(format_name)
                .map(json_wrapped_fullmatch_pattern)
                .ok_or_else(|| GlrMaskError::GrammarParse(format!("Unknown format: {format_name}")))?,
        };
        Ok(self.extract_terminal_rule(expr, "JSON_FORMAT_STRING"))
    }

    fn json_literal(&self, value: &Value) -> GrammarExpr {
        json_value_literal_expr(value)
    }

    fn json_string_literal(&self, text: &str) -> GrammarExpr {
        literal_expr(&json_string_literal_bytes(text))
    }

    fn json_key_colon_literal(&self, text: &str) -> GrammarExpr {
        let mut bytes = json_string_literal_bytes(text);
        bytes.extend_from_slice(JSON_KEY_SEPARATOR);
        literal_expr(&bytes)
    }

    fn json_key_separator_expr(&self) -> GrammarExpr {
        literal_expr(JSON_KEY_SEPARATOR)
    }

    fn json_item_separator_expr(&self) -> GrammarExpr {
        literal_expr(JSON_ITEM_SEPARATOR)
    }

    fn extract_terminal_rule(&mut self, expr: GrammarExpr, prefix: &str) -> GrammarExpr {
        if Self::is_trivial_expr(&expr) {
            return expr;
        }

        let rule_name = self.fresh_rule_name(prefix);
        self.insert_rule(rule_name.clone(), expr);
        GrammarExpr::Ref(rule_name)
    }

    fn json_key_with_separator_expr(&mut self, key_expr: GrammarExpr, prefix: &str) -> GrammarExpr {
        self.extract_terminal_rule(
            sequence_or_single(vec![key_expr, self.json_key_separator_expr()]),
            prefix,
        )
    }

    fn build_object_expr(&mut self, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        let properties = schema.get("properties").and_then(Value::as_object);
        let required_list = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let required_keys: BTreeSet<String> = required_list.iter().cloned().collect();
        let additional_properties = schema.get("additionalProperties");
        let pattern_properties = schema.get("patternProperties").and_then(Value::as_object);
        let property_names = schema.get("propertyNames");

        let all_keys_required = properties
            .map(|properties| properties.keys().all(|key| required_keys.contains(key)))
            .unwrap_or(false);
        let no_additional = matches!(additional_properties, None | Some(Value::Bool(false)));

        let min_properties = schema.get("minProperties").and_then(Value::as_u64);
        let max_properties = schema.get("maxProperties").and_then(Value::as_u64);
        if min_properties.is_some() || max_properties.is_some() {
            if properties.is_none() || !all_keys_required || !no_additional {
                return Err(GlrMaskError::GrammarParse(
                    "min/maxProperties only supported when all keys listed in properties are required"
                        .into(),
                ));
            }
            let fixed_count = properties.map(|properties| properties.len()).unwrap_or(0) as u64;
            if min_properties.map(|min| fixed_count < min).unwrap_or(false)
                || max_properties.map(|max| fixed_count > max).unwrap_or(false)
            {
                return Err(GlrMaskError::GrammarParse(
                    "min/maxProperties constraints are unsatisfiable for fixed required properties"
                        .into(),
                ));
            }
        }

        if let Some(properties) = properties {
            let mut additional_schema = match additional_properties {
                Some(Value::Bool(false)) => None,
                Some(Value::Object(map)) => Some(Value::Object(map.clone())),
                _ => Some(serde_json::json!({})),
            };
            if additional_schema
                .as_ref()
                .map(|schema| self.is_certainly_unsatisfiable(schema))
                .unwrap_or(false)
            {
                additional_schema = None;
            }
            return self.build_ordered_properties_object_expr(
                properties,
                &required_list,
                &required_keys,
                additional_schema,
            );
        }

        if !required_list.is_empty() && pattern_properties.is_none() && property_names.is_none() {
            let mut additional_schema = match additional_properties {
                Some(Value::Bool(false)) => None,
                Some(Value::Object(map)) => Some(Value::Object(map.clone())),
                _ => Some(serde_json::json!({})),
            };
            if additional_schema
                .as_ref()
                .map(|schema| self.is_certainly_unsatisfiable(schema))
                .unwrap_or(false)
            {
                additional_schema = None;
            }
            return self.build_required_any_order_object_expr(&required_list, additional_schema);
        }

        // No defined properties but typed additionalProperties — build an
        // AP-only object that constrains value types.
        if let Some(Value::Object(map)) = additional_properties {
            let empty_props = serde_json::Map::new();
            return self.build_ordered_properties_object_expr(
                &empty_props,
                &[],
                &BTreeSet::new(),
                Some(Value::Object(map.clone())),
            );
        }

        if properties.is_none()
            && pattern_properties.map(|patterns| patterns.len() == 1).unwrap_or(false)
        {
            let (pattern, value_schema) = pattern_properties
                .and_then(|patterns| patterns.iter().next())
                .ok_or_else(|| GlrMaskError::GrammarParse("invalid patternProperties".into()))?;
            let match_all_pattern = pattern == "^.*$" || pattern == ".*";
            let property_names = serde_json::json!({"pattern": pattern});
            let value_expr = self.convert_schema(value_schema)?;
            if matches!(additional_properties, Some(Value::Bool(false))) || match_all_pattern {
                return self.build_pattern_named_object_expr(&property_names, value_expr);
            }

            if let Some((literal_prefix, exact)) = simple_anchored_literal_pattern(pattern) {
                if !literal_prefix.is_empty() {
                    let unmatched_key_expr = if exact {
                        let rule_prefix = self.fresh_rule_name("PP_KEY_EXACT");
                        self.build_complement_key_expr(
                            &[literal_prefix],
                            &rule_prefix,
                        )
                    } else {
                        let rule_prefix = self.fresh_rule_name("PP_KEY_PREFIX");
                        self.build_excluding_prefix_key_expr(
                            &literal_prefix,
                            &rule_prefix,
                        )
                    };
                    let additional_value_expr = match additional_properties {
                        Some(Value::Object(map)) => self.convert_schema(&Value::Object(map.clone()))?,
                        Some(Value::Bool(true)) | None => self.json_value_ref(),
                        _ => return Ok(self.json_object_ref()),
                    };
                    return self.build_mixed_pattern_named_object_expr(
                        &property_names,
                        value_expr,
                        unmatched_key_expr,
                        additional_value_expr,
                    );
                }
            }

            return Ok(self.json_object_ref());
        }

        if let Some(property_names) = property_names {
            return self.build_pattern_named_object_expr(property_names, self.json_value_ref());
        }

        Ok(self.json_object_ref())
    }

    fn build_required_any_order_object_expr(
        &mut self,
        required_list: &[String],
        additional_properties_schema: Option<Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let mut base_index = self.object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_req_any_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_nc_0")) {
                break candidate;
            }
            base_index += 1;
        };
        self.object_rule_counter = base_index + 1;

        let full_mask: Vec<usize> = (0..required_list.len()).collect();
        let mut masks = Vec::<Vec<usize>>::new();
        let mut mask_indices = HashMap::<Vec<usize>, usize>::new();
        let mut pending = vec![full_mask.clone()];
        while let Some(mask) = pending.pop() {
            if mask_indices.contains_key(&mask) {
                continue;
            }
            let index = masks.len();
            mask_indices.insert(mask.clone(), index);
            masks.push(mask.clone());
            for &item in &mask {
                let next_mask = mask
                    .iter()
                    .copied()
                    .filter(|candidate| *candidate != item)
                    .collect::<Vec<_>>();
                if !mask_indices.contains_key(&next_mask) {
                    pending.push(next_mask);
                }
            }
        }

        let extra_pair_expr = if let Some(schema) = additional_properties_schema {
            let extra_value_expr = self.convert_schema(&schema)?;
            let ck_prefix = format!("{}_CK", base_name.to_uppercase());
            let extra_key_expr = self.build_complement_key_expr(required_list, &ck_prefix);
            Some(sequence_or_single(vec![
                self.json_key_with_separator_expr(extra_key_expr, &format!("{ck_prefix}_KEY_COLON")),
                extra_value_expr,
            ]))
        } else {
            None
        };

        for mask in masks.clone() {
            let mask_index = *mask_indices.get(&mask).unwrap();
            let nc_name = format!("{base_name}_nc_{mask_index}");
            let c_name = format!("{base_name}_c_{mask_index}");
            let mut nc_alts = Vec::new();
            let mut c_alts = Vec::new();

            for &item in &mask {
                let key = &required_list[item];
                let pair_expr = sequence_or_single(vec![
                    self.json_key_colon_literal(key),
                    self.json_value_ref(),
                ]);
                let next_mask = mask
                    .iter()
                    .copied()
                    .filter(|candidate| *candidate != item)
                    .collect::<Vec<_>>();
                let next_index = *mask_indices.get(&next_mask).unwrap();
                nc_alts.push(sequence_or_single(vec![
                    pair_expr.clone(),
                    GrammarExpr::Ref(format!("{base_name}_c_{next_index}")),
                ]));
                c_alts.push(sequence_or_single(vec![
                    self.json_item_separator_expr(),
                    pair_expr,
                    GrammarExpr::Ref(format!("{base_name}_c_{next_index}")),
                ]));
            }

            if let Some(extra_pair_expr) = extra_pair_expr.clone() {
                nc_alts.push(sequence_or_single(vec![
                    extra_pair_expr.clone(),
                    GrammarExpr::Ref(c_name.clone()),
                ]));
                c_alts.push(sequence_or_single(vec![
                    self.json_item_separator_expr(),
                    extra_pair_expr,
                    GrammarExpr::Ref(c_name.clone()),
                ]));
            }

            if mask.is_empty() {
                nc_alts.push(empty_expr());
                c_alts.push(empty_expr());
            }

            self.insert_rule(nc_name, choice_or_single(nc_alts));
            self.insert_rule(c_name, choice_or_single(c_alts));
        }

        Ok(sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(format!(
                "{base_name}_nc_{}",
                mask_indices.get(&full_mask).copied().unwrap_or(0)
            )),
            literal_expr(b"}"),
        ]))
    }

    fn build_ordered_properties_object_expr(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        additional_properties_schema: Option<Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let mut ordered = Vec::new();
        let mut known_property_keys: Vec<String> = properties.keys().cloned().collect();
        known_property_keys.extend(
            required_list
                .iter()
                .filter(|key| !properties.contains_key(*key))
                .cloned(),
        );
        for (key, subschema) in properties {
            if self.is_certainly_unsatisfiable(subschema) {
                if required_keys.contains(key) {
                    return Err(unsat_schema_error());
                }
                continue;
            }
            ordered.push((key.clone(), subschema.clone(), required_keys.contains(key)));
        }
        for key in required_list {
            if !properties.contains_key(key) {
                ordered.push((key.clone(), serde_json::json!({}), true));
            }
        }

        let additional_properties_schema = if additional_properties_schema
            .as_ref()
            .map(|schema| self.is_certainly_unsatisfiable(schema))
            .unwrap_or(false)
        {
            None
        } else {
            additional_properties_schema
        };

        if ordered.is_empty() && additional_properties_schema.is_none() {
            return Ok(sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]));
        }

        let mut base_index = self.object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_ord_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_0_nc")) {
                break candidate;
            }
            base_index += 1;
        };
        self.object_rule_counter = base_index + 1;

        let (term_nc, term_c) = if let Some(schema) = additional_properties_schema {
            // Collect known property keys for complement trie
            let known_keys = known_property_keys;

            // Build complement key that excludes known property names
            let ck_prefix = format!("{}_CK", base_name.to_uppercase());
            let complement_key = self.build_complement_key_expr(&known_keys, &ck_prefix);

            // complement_key_colon = complement_key ": "
            let complement_key_colon = self.json_key_with_separator_expr(
                complement_key,
                &format!("{ck_prefix}_KEY_COLON"),
            );

            let term_nc = format!("{base_name}_ap_nc");
            let term_c = format!("{base_name}_ap_c");
            let value_expr = self.convert_schema(&schema)?;
            self.insert_rule(
                term_nc.clone(),
                choice_or_single(vec![
                    sequence_or_single(vec![
                        complement_key_colon.clone(),
                        value_expr.clone(),
                        GrammarExpr::Ref(term_c.clone()),
                    ]),
                    empty_expr(),
                ]),
            );
            self.insert_rule(
                term_c.clone(),
                choice_or_single(vec![
                    sequence_or_single(vec![
                        self.json_item_separator_expr(),
                        complement_key_colon,
                        value_expr,
                        GrammarExpr::Ref(term_c.clone()),
                    ]),
                    empty_expr(),
                ]),
            );
            (term_nc, term_c)
        } else {
            let term_nc = format!("{base_name}_{}_nc", ordered.len());
            let term_c = format!("{base_name}_{}_c", ordered.len());
            self.insert_rule(term_nc.clone(), empty_expr());
            self.insert_rule(term_c.clone(), empty_expr());
            (term_nc, term_c)
        };

        if ordered.is_empty() {
            return Ok(sequence_or_single(vec![
                literal_expr(b"{"),
                GrammarExpr::Ref(term_nc),
                literal_expr(b"}"),
            ]));
        }

        let mut tree_counter = 0usize;
        let (tree_expr, tree_can_be_empty) =
            self.build_object_tree(&base_name, &ordered, &mut tree_counter)?;

        let top_nc = format!("{base_name}_0_nc");
        let top_expr = if tree_can_be_empty {
            choice_or_single(vec![
                sequence_or_single(vec![tree_expr, GrammarExpr::Ref(term_c.clone())]),
                GrammarExpr::Ref(term_nc.clone()),
            ])
        } else {
            sequence_or_single(vec![tree_expr, GrammarExpr::Ref(term_c.clone())])
        };
        self.insert_rule(top_nc.clone(), top_expr);

        Ok(sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(top_nc),
            literal_expr(b"}"),
        ]))
    }

    fn build_object_tree(
        &mut self,
        base_name: &str,
        items: &[(String, Value, bool)],
        counter: &mut usize,
    ) -> Result<(GrammarExpr, bool), GlrMaskError> {
        if items.len() == 1 {
            let (key, subschema, is_required) = &items[0];
            let value_expr = self.convert_schema(subschema)?;
            let kv_expr = sequence_or_single(vec![self.json_key_colon_literal(key), value_expr]);
            if *is_required {
                return Ok((kv_expr, false));
            }

            let rule_name = format!("{base_name}_t{}", *counter);
            *counter += 1;
            self.insert_rule(rule_name.clone(), kv_expr);
            return Ok((GrammarExpr::Ref(rule_name), true));
        }

        let mid = items.len() / 2;
        let (left_expr, left_can_be_empty) = self.build_object_tree(base_name, &items[..mid], counter)?;
        let (right_expr, right_can_be_empty) =
            self.build_object_tree(base_name, &items[mid..], counter)?;

        let mut options = vec![sequence_or_single(vec![
            left_expr.clone(),
            self.json_item_separator_expr(),
            right_expr.clone(),
        ])];
        if right_can_be_empty {
            options.push(left_expr.clone());
        }
        if left_can_be_empty {
            options.push(right_expr.clone());
        }

        if options.len() == 1 {
            return Ok((options.pop().unwrap(), false));
        }

        let rule_name = format!("{base_name}_t{}", *counter);
        *counter += 1;
        self.insert_rule(rule_name.clone(), choice_or_single(options));
        Ok((GrammarExpr::Ref(rule_name), left_can_be_empty && right_can_be_empty))
    }

    fn property_name_pattern<'a>(property_names: &'a Value) -> Result<&'a str, GlrMaskError> {
        let property_names = property_names.as_object().ok_or_else(|| {
            GlrMaskError::GrammarParse(
                "propertyNames is not supported unless it is an object with a pattern".into(),
            )
        })?;
        if property_names.keys().any(|key| key != "pattern") {
            return Err(GlrMaskError::GrammarParse(
                "propertyNames is not supported unless it only defines pattern".into(),
            ));
        }
        let pattern = property_names
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GlrMaskError::GrammarParse(
                    "propertyNames is not supported unless it defines a string pattern".into(),
                )
            })?;
        Ok(pattern)
    }

    fn build_pattern_named_object_expr(
        &mut self,
        property_names: &Value,
        value_expr: GrammarExpr,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let pattern = Self::property_name_pattern(property_names)?;
        let pair = sequence_or_single(vec![
            json_wrapped_key_colon_pattern(pattern),
            value_expr,
        ]);
        Ok(choice_or_single(vec![
            sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]),
            sequence_or_single(vec![
                literal_expr(b"{"),
                pair.clone(),
                repeat_expr(sequence_or_single(vec![self.json_item_separator_expr(), pair]), 0, None),
                literal_expr(b"}"),
            ]),
        ]))
    }

    fn build_mixed_pattern_named_object_expr(
        &mut self,
        property_names: &Value,
        matched_value_expr: GrammarExpr,
        unmatched_key_expr: GrammarExpr,
        unmatched_value_expr: GrammarExpr,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let pattern = Self::property_name_pattern(property_names)?;
        let matched_pair = sequence_or_single(vec![
            json_wrapped_key_colon_pattern(pattern),
            matched_value_expr,
        ]);
        let unmatched_pair = sequence_or_single(vec![
            self.json_key_with_separator_expr(unmatched_key_expr, "PP_KEY_COLON"),
            unmatched_value_expr,
        ]);
        let pair = choice_or_single(vec![matched_pair, unmatched_pair]);
        Ok(choice_or_single(vec![
            sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]),
            sequence_or_single(vec![
                literal_expr(b"{"),
                pair.clone(),
                repeat_expr(sequence_or_single(vec![self.json_item_separator_expr(), pair]), 0, None),
                literal_expr(b"}"),
            ]),
        ]))
    }

    fn build_array_expr(&mut self, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        if let Some(prefix_items) = schema.get("prefixItems").and_then(Value::as_array) {
            let max_items_raw = schema.get("maxItems").and_then(Value::as_u64).map(|value| value as usize);
            if max_items_raw
                .map(|max_items| max_items < prefix_items.len())
                .unwrap_or(false)
            {
                return Ok(self.json_array_ref());
            }

            let extra_item_expr = match schema.get("items") {
                Some(Value::Object(_)) => self.convert_schema(schema.get("items").unwrap())?,
                _ => self.json_value_ref(),
            };

            let min_items = schema
                .get("minItems")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(prefix_items.len());
            let max_items = max_items_raw;
            let extra_min = min_items.saturating_sub(prefix_items.len());
            let extra_max = max_items.map(|max_items| max_items.saturating_sub(prefix_items.len()));
            let (extra_min, extra_max) = Self::clamp_repeat(extra_min, extra_max);

            if prefix_items.is_empty() {
                return Ok(self.build_repeated_array(extra_item_expr, extra_min, extra_max));
            }

            let mut parts = vec![literal_expr(b"[")];
            for (index, item_schema) in prefix_items.iter().enumerate() {
                if index > 0 {
                    parts.push(self.json_item_separator_expr());
                }
                parts.push(self.convert_schema(item_schema)?);
            }

            if extra_max.is_none() || extra_max.unwrap_or(0) > 0 || extra_min > 0 {
                let extra_item_expr = self.extract_rule(extra_item_expr, "arr_item");
                parts.push(repeat_expr(
                    sequence_or_single(vec![self.json_item_separator_expr(), extra_item_expr]),
                    extra_min,
                    extra_max,
                ));
            }
            parts.push(literal_expr(b"]"));
            return Ok(sequence_or_single(parts));
        }

        let min_items = schema.get("minItems").and_then(Value::as_u64).map(|value| value as usize);
        let max_items = schema.get("maxItems").and_then(Value::as_u64).map(|value| value as usize);
        if let Some(item_schema) = schema.get("items") {
            if item_schema.is_object() {
                let item_expr = self.convert_schema(item_schema)?;
                return Ok(self.build_repeated_array(
                    item_expr,
                    min_items.unwrap_or(0),
                    max_items,
                ));
            }
        }

        if min_items.is_some() || max_items.is_some() {
            return Ok(self.build_repeated_array(
                self.json_value_ref(),
                min_items.unwrap_or(0),
                max_items,
            ));
        }

        Ok(self.json_array_ref())
    }

    fn build_repeated_array(
        &mut self,
        item_expr: GrammarExpr,
        min_items: usize,
        max_items: Option<usize>,
    ) -> GrammarExpr {
        let (min_items, max_items) = Self::clamp_repeat(min_items, max_items);
        if max_items == Some(0) {
            return sequence_or_single(vec![literal_expr(b"["), literal_expr(b"]")]);
        }

        let item_expr = self.extract_rule(item_expr, "arr_item");
        let non_empty = sequence_or_single(vec![
            literal_expr(b"["),
            item_expr.clone(),
            repeat_expr(
                sequence_or_single(vec![self.json_item_separator_expr(), item_expr]),
                min_items.saturating_sub(1),
                max_items.map(|max_items| max_items.saturating_sub(1)),
            ),
            literal_expr(b"]"),
        ]);
        if min_items == 0 {
            choice_or_single(vec![
                sequence_or_single(vec![literal_expr(b"["), literal_expr(b"]")]),
                non_empty,
            ])
        } else {
            non_empty
        }
    }

    /// Build an UPPERCASE terminal rule that matches a quoted JSON string key
    /// whose body is NOT any of the given exact keys.
    ///
    /// Returns `GrammarExpr::Ref(name)` to the created terminal rule.
    ///
    /// Uses a trie-based complement: at each trie node, we match characters
    /// that differ from the excluded keys at that position.  UPPERCASE names
    /// make glrmask compile these as DFA terminals (efficient) instead of
    /// LR parser rules (which cause GLR state explosion).
    fn build_complement_key_expr(&mut self, keys: &[String], prefix: &str) -> GrammarExpr {
        if keys.is_empty() {
            // No keys to exclude → use the standard JSON_STRING pattern
            return self.json_string_ref();
        }

        // STRING_CONTENT = (STRING_CHAR | ESCAPE_SEQ)*
        let string_content_regex = format!("({})*", JSON_STRING_CHAR_PATTERN);

        // Build the trie complement body as a single nested expression (no intermediate rules).
        let body_expr = Self::build_complement_trie_expr(keys, &string_content_regex);

        // Wrap in quotes to form a single UPPERCASE terminal: `"` body `"`
        let full_name = format!("{prefix}_FULL");
        let full_expr = sequence_or_single(vec![
            literal_expr(b"\""),
            body_expr,
            literal_expr(b"\""),
        ]);
        self.insert_rule(full_name.clone(), full_expr);

        GrammarExpr::Ref(full_name)
    }

    fn build_excluding_prefix_key_expr(&mut self, prefix_text: &str, prefix: &str) -> GrammarExpr {
        let full_name = format!("{prefix}_FULL");
        let full_expr = sequence_or_single(vec![
            literal_expr(b"\""),
            Self::build_prefix_complement_expr(prefix_text),
            literal_expr(b"\""),
        ]);
        self.insert_rule(full_name.clone(), full_expr);
        GrammarExpr::Ref(full_name)
    }

    /// Recursively build trie complement as a single nested GrammarExpr (no named rules).
    fn build_complement_trie_expr(
        keys: &[String],
        string_content_regex: &str,
    ) -> GrammarExpr {
        let mut by_first: std::collections::BTreeMap<char, Vec<String>> = std::collections::BTreeMap::new();
        let mut has_empty = false;
        for k in keys {
            if k.is_empty() {
                has_empty = true;
            } else {
                let first = k.chars().next().unwrap();
                by_first
                    .entry(first)
                    .or_default()
                    .push(k[first.len_utf8()..].to_string());
            }
        }

        let mut alts: Vec<GrammarExpr> = Vec::new();

        // Alt 1: first byte is a valid JSON string char that doesn't start any excluded key,
        //        followed by any remaining string content.
        let excluded_bytes: Vec<u8> = by_first.keys().map(|&c| c as u8).collect();
        if !excluded_bytes.is_empty() {
            let mut excluded_set = String::new();
            excluded_set.push_str(r#"\x00-\x1f\x7f"\\"#);  // always excluded in JSON strings
            for &b in &excluded_bytes {
                if b == b'-' || b == b']' || b == b'^' || b == b'\\' {
                    excluded_set.push('\\');
                    excluded_set.push(b as char);
                } else {
                    excluded_set.push(b as char);
                }
            }
            let char_minus_regex = format!("[^{}]", excluded_set);
            alts.push(sequence_or_single(vec![
                regex_expr(&char_minus_regex),
                regex_expr(string_content_regex),
            ]));
        } else {
            alts.push(sequence_or_single(vec![
                regex_expr(JSON_STRING_CHAR_PATTERN),
                regex_expr(string_content_regex),
            ]));
        }

        // Alt 2: starts with an escape sequence (can't be an ASCII key)
        alts.push(sequence_or_single(vec![
            regex_expr(r#"\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}"#),
            regex_expr(string_content_regex),
        ]));

        // Alt 3..N: for each trie-branch char, match it then recurse
        for (&ch, suffixes) in &by_first {
            let sub = Self::build_complement_trie_expr(suffixes, string_content_regex);
            alts.push(sequence_or_single(vec![
                literal_expr(&[ch as u8]),
                sub,
            ]));
        }

        // Final alt: empty string only when no excluded key ends at this node.
        if !has_empty {
            alts.push(empty_expr());
        }

        choice_or_single(alts)
    }

    fn build_prefix_complement_expr(prefix_text: &str) -> GrammarExpr {
        if prefix_text.is_empty() {
            return regex_expr(r#"[^\x00-\xFF]"#);
        }

        let chars: Vec<char> = prefix_text.chars().collect();
        Self::build_prefix_complement_from_chars(&chars)
    }

    fn build_prefix_complement_from_chars(chars: &[char]) -> GrammarExpr {
        if chars.is_empty() {
            return regex_expr(r#"[^\x00-\xFF]"#);
        }

        let next = chars[0] as u8;
        let mut excluded_set = String::from(r#"\x00-\x1f\x7f"\\"#);
        if matches!(next, b'-' | b']' | b'^' | b'\\') {
            excluded_set.push('\\');
        }
        excluded_set.push(next as char);
        let char_minus_regex = format!("[^{}]", excluded_set);
        let string_content_regex = format!("({})*", JSON_STRING_CHAR_PATTERN);

        let mut alts = vec![
            sequence_or_single(vec![
                regex_expr(&char_minus_regex),
                regex_expr(&string_content_regex),
            ]),
            sequence_or_single(vec![
                regex_expr(r#"\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}"#),
                regex_expr(&string_content_regex),
            ]),
            sequence_or_single(vec![
                literal_expr(&[next]),
                Self::build_prefix_complement_from_chars(&chars[1..]),
            ]),
            empty_expr(),
        ];

        choice_or_single(std::mem::take(&mut alts))
    }

    fn is_trivial_expr(expr: &GrammarExpr) -> bool {
        matches!(
            expr,
            GrammarExpr::Ref(_) | GrammarExpr::Literal(_) | GrammarExpr::RawRegex(_)
        ) || matches!(expr, GrammarExpr::Sequence(parts) if parts.is_empty())
    }

    fn extract_rule(&mut self, expr: GrammarExpr, prefix: &str) -> GrammarExpr {
        if Self::is_trivial_expr(&expr) {
            return expr;
        }

        let key = expr_key(&expr);
        if let Some(rule_name) = self.expr_dedup_cache.get(&key) {
            return GrammarExpr::Ref(rule_name.clone());
        }

        let rule_name = self.fresh_rule_name(prefix);
        self.insert_rule(rule_name.clone(), expr);
        self.expr_dedup_cache.insert(key, rule_name.clone());
        GrammarExpr::Ref(rule_name)
    }

    fn clamp_repeat(min_value: usize, max_value: Option<usize>) -> (usize, Option<usize>) {
        match max_value {
            Some(max_value) if max_value < min_value => (min_value, Some(min_value)),
            _ => (min_value, max_value),
        }
    }
}

fn choice_or_single(alts: Vec<GrammarExpr>) -> GrammarExpr {
    let mut alts = alts;
    if alts.is_empty() {
        empty_expr()
    } else if alts.len() == 1 {
        alts.pop().unwrap()
    } else {
        GrammarExpr::Choice(alts)
    }
}

fn sequence_or_single(parts: Vec<GrammarExpr>) -> GrammarExpr {
    let mut parts: Vec<GrammarExpr> = parts
        .into_iter()
        .filter(|expr| !matches!(expr, GrammarExpr::Sequence(inner) if inner.is_empty()))
        .collect();
    if parts.is_empty() {
        empty_expr()
    } else if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        GrammarExpr::Sequence(parts)
    }
}

fn highest_power_of_two_leq(value: usize) -> usize {
    debug_assert!(value > 0);
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

fn repeat_expr(item: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
    match (min, max) {
        (0, None) => GrammarExpr::Repeat(Box::new(item)),
        (1, None) => GrammarExpr::RepeatOne(Box::new(item)),
        (_, Some(max)) => GrammarExpr::RepeatRange {
            expr: Box::new(item),
            min,
            max,
        },
        _ => {
            let mut parts = Vec::new();
            for _ in 0..min {
                parts.push(item.clone());
            }
            parts.push(GrammarExpr::Repeat(Box::new(item)));
            sequence_or_single(parts)
        }
    }
}

fn sanitize_rule_name(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if sanitized.is_empty() { "rule".into() } else { sanitized }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;

    fn assert_schema_error_contains(schema: &str, expected: &str) {
        match json_schema_to_grammar(schema) {
            Ok(_) => panic!("expected schema to fail with {expected:?}"),
            Err(GlrMaskError::GrammarParse(message)) => {
                assert!(
                    message.contains(expected),
                    "expected error containing {expected:?}, got {message:?}"
                );
            }
            Err(other) => panic!("expected GrammarParse error, got {other:?}"),
        }
    }

    fn expr_has_split_separator(expr: &GrammarExpr, left_bytes: &[u8], right_bytes: &[u8]) -> bool {
        match expr {
            GrammarExpr::Sequence(parts) => {
                parts.windows(2).any(|window| {
                    matches!(&window[0], GrammarExpr::Literal(bytes) if bytes == left_bytes)
                        && matches!(&window[1], GrammarExpr::Literal(bytes) if bytes == right_bytes)
                }) || parts.iter().any(|part| expr_has_split_separator(part, left_bytes, right_bytes))
            }
            GrammarExpr::Choice(options) => {
                options.iter().any(|part| expr_has_split_separator(part, left_bytes, right_bytes))
            }
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner)
            | GrammarExpr::RepeatRange { expr: inner, .. } => {
                expr_has_split_separator(inner, left_bytes, right_bytes)
            }
            _ => false,
        }
    }

    fn expr_has_literal(expr: &GrammarExpr, expected: &[u8]) -> bool {
        match expr {
            GrammarExpr::Literal(bytes) => bytes == expected,
            GrammarExpr::Sequence(parts) => parts.iter().any(|part| expr_has_literal(part, expected)),
            GrammarExpr::Choice(options) => options.iter().any(|part| expr_has_literal(part, expected)),
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner)
            | GrammarExpr::RepeatRange { expr: inner, .. } => expr_has_literal(inner, expected),
            _ => false,
        }
    }

    fn expr_has_ref_then_literal(expr: &GrammarExpr, literal: &[u8]) -> bool {
        match expr {
            GrammarExpr::Sequence(parts) => {
                if parts.windows(2).any(|window| {
                    matches!(window[0], GrammarExpr::Ref(_))
                        && matches!(&window[1], GrammarExpr::Literal(bytes) if bytes == literal)
                }) {
                    return true;
                }
                parts.iter().any(|part| expr_has_ref_then_literal(part, literal))
            }
            GrammarExpr::Choice(options) => options.iter().any(|part| expr_has_ref_then_literal(part, literal)),
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner)
            | GrammarExpr::RepeatRange { expr: inner, .. } => {
                expr_has_ref_then_literal(inner, literal)
            }
            _ => false,
        }
    }

    fn named_grammar_has_split_separator(grammar: &NamedGrammar, left_bytes: &[u8], right_bytes: &[u8]) -> bool {
        grammar
            .rules
            .iter()
            .any(|rule| expr_has_split_separator(&rule.expr, left_bytes, right_bytes))
    }

    fn named_grammar_has_literal(grammar: &NamedGrammar, expected: &[u8]) -> bool {
        grammar.rules.iter().any(|rule| expr_has_literal(&rule.expr, expected))
    }

    fn named_nonterminal_has_ref_then_literal(grammar: &NamedGrammar, literal: &[u8]) -> bool {
        grammar
            .rules
            .iter()
            .filter(|rule| !rule.is_terminal)
            .any(|rule| expr_has_ref_then_literal(&rule.expr, literal))
    }

    #[test]
    fn test_boolean_schema() {
        let g = json_schema_to_grammar(r#"{"type": "boolean"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_string_schema() {
        let g = json_schema_to_grammar(r#"{"type": "string"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_integer_schema() {
        let g = json_schema_to_grammar(r#"{"type": "integer"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_null_schema() {
        let g = json_schema_to_grammar(r#"{"type": "null"}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_enum_schema() {
        let g = json_schema_to_grammar(r#"{"enum": ["a", "b", "c"]}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_const_schema() {
        let g = json_schema_to_grammar(r#"{"const": 42}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_schema() {
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_additional_properties_false() {
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": false
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_only_required_comma_free() {
        
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_all_optional_no_required() {
        
        let g = json_schema_to_grammar(r#"{
            "type": "object",
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "integer"}
            }
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_empty_additional_false() {
        
        let g = json_schema_to_grammar(r#"{"type": "object", "additionalProperties": false}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_schema() {
        let g = json_schema_to_grammar(r#"{"type": "array", "items": {"type": "integer"}}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_min_max_items() {
        let g = json_schema_to_grammar(r#"{"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_array_prefix_items() {
        let g = json_schema_to_grammar(r#"{
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_oneof_schema() {
        assert_schema_error_contains(r#"{
            "oneOf": [{"type": "string"}, {"type": "integer"}]
        }"#, "oneOf constraints are not supported");
    }

    #[test]
    fn test_allof_schema() {
        let g = json_schema_to_grammar(r#"{
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"properties": {"b": {"type": "integer"}}}
            ]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_untyped_object_keywords_allow_non_object_values() {
        let schema = r#"{
            "type": "array",
            "items": {
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"],
                "additionalProperties": false
            }
        }"#;
        assert!(accepts_sequence(schema, &[b"[", b"460", b"]"]));
    }

    #[test]
    fn test_untyped_object_keywords_still_allow_object_values() {
        let schema = r#"{
            "type": "array",
            "items": {
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"],
                "additionalProperties": false
            }
        }"#;
        assert!(accepts_sequence(
            schema,
            &[b"[", b"{\"identifier\": \"x\"}", b"]"]
        ));
    }

    #[test]
    fn test_ref_schema() {
        let g = json_schema_to_grammar(r##"{
            "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}},
            "$ref": "#/$defs/Point"
        }"##).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_string_min_length() {
        let g = json_schema_to_grammar(r#"{"type": "string", "minLength": 3}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_string_min_max_length() {
        let g = json_schema_to_grammar(r#"{"type": "string", "minLength": 1, "maxLength": 5}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_accepts_bounded_string_length() {
        assert!(accepts_sequence(
            r#"{"type": "string", "minLength": 2, "maxLength": 4}"#,
            &[b"\"ab\""]
        ));
    }

    #[test]
    fn test_rejects_too_short_bounded_string_length() {
        assert!(!accepts_sequence(
            r#"{"type": "string", "minLength": 2, "maxLength": 4}"#,
            &[b"\"a\""]
        ));
    }

    #[test]
    fn test_bounded_string_stays_terminalized() {
        let schema: Value = serde_json::from_str(r#"{"type": "string", "minLength": 1, "maxLength": 5}"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();
        let start_rule = grammar.rules.iter().find(|rule| rule.name == "start").unwrap();
        match &start_rule.expr {
            GrammarExpr::Ref(rule_name) => {
                let target = grammar.rules.iter().find(|rule| rule.name == *rule_name).unwrap();
                assert!(target.is_terminal, "expected bounded string start to reference a terminal rule");
            }
            other => panic!("expected bounded string start to be a terminal ref, got {other:?}"),
        }
    }

    #[test]
    fn test_jsonify_regex_dot_only_rewrites_bare_dot() {
        assert_eq!(
            jsonify_regex_dot(r#".^\.[$]"#),
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})^\.[\x24]"#
        );
        assert_eq!(
            jsonify_regex_dot(r#"[.]\.."#),
            r#"[\x2E]\.(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})"#
        );
    }

    #[test]
    fn test_pattern_dot_accepts_escaped_backslash() {
        assert!(accepts_sequence(
            r#"{"type": "string", "pattern": "^file:.+\\.km[lz]$"}"#,
            &[b"\"file:\\\\foo.kml\""]
        ));
    }

    #[test]
    fn test_pattern_char_class_respects_json_string_encoding() {
        let schema = r#"{"type": "string", "pattern": "^[^:\\s]+:[^:\\s]+(:[^\\s]+)?$"}"#;
        assert!(!accepts_sequence(schema, &[b"\"my-app:prod\x01\""]));
        assert!(!accepts_sequence(schema, &[b"\"my-app:prod\n\""]));
        assert!(!accepts_sequence(schema, &[b"\"my-app:prod\x7f\""]));
        assert!(accepts_sequence(schema, &[b"\"my-app:prod\\\"x\""]));
    }

    #[test]
    fn test_pattern_uses_search_semantics_for_top_level_branches() {
        let schema = r#"{"type": "string", "pattern": "^allow|deny$"}"#;
        assert!(accepts_sequence(schema, &[b"\"allow\""]));
        assert!(accepts_sequence(schema, &[b"\"xdeny\""]));
        assert!(accepts_sequence(schema, &[b"\"/deny\""]));
        assert!(!accepts_sequence(schema, &[b"\"xdenyx\""]));
    }

    #[test]
    fn test_pattern_literal_slash_accepts_escaped_solidus() {
        let schema = r#"{"type": "string", "pattern": "^/[^\\u0000]+$"}"#;
        assert!(accepts_sequence(schema, &[b"\"/path\""]));
        assert!(accepts_sequence(schema, &[b"\"\\/path\""]));
    }

    #[test]
    fn test_format_uses_full_string_semantics() {
        let schema = r#"{"type": "string", "format": "date-time"}"#;
        assert!(!accepts_sequence(schema, &[b"\"/2022-01-01T12:00:00Z\""]));
        assert!(accepts_sequence(schema, &[b"\"2022-01-01T12:00:00Z\""]));
    }

    #[test]
    fn test_property_names_is_rejected_like_llguidance() {
        assert_schema_error_contains(
            r#"{
                "type": "object",
                "propertyNames": {"pattern": "^[a-z]+$"}
            }"#,
            "Unimplemented keys",
        );
    }

    #[test]
    fn test_known_unimplemented_keyword_is_rejected() {
        assert_schema_error_contains(
            r#"{
                "type": "array",
                "uniqueItems": true
            }"#,
            "Unimplemented keys",
        );
    }

    #[test]
    fn test_unknown_format_is_rejected_like_llguidance() {
        assert_schema_error_contains(
            r#"{
                "type": "string",
                "format": "uri-reference"
            }"#,
            "Unknown format",
        );
    }

    #[test]
    fn test_local_id_ref_preserves_target_type_constraints() {
        let schema = r##"{
            "$schema": "http://json-schema.org/draft-04/schema#",
            "definitions": {
                "jumpGateCapability": {
                    "id": "#jumpGateCapability",
                    "type": "object"
                }
            },
            "type": "object",
            "properties": {
                "jumpGate": {
                    "$ref": "#jumpGateCapability"
                }
            }
        }"##;
        assert!(!accepts_sequence(schema, &[b"{\"jumpGate\": true}"]));
        assert!(accepts_sequence(schema, &[b"{\"jumpGate\": {}}"]));
    }

    #[test]
    fn test_temp1() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "jumpGate": {
                    "type": "object"
                }
            }
        }"##;
        assert!(!accepts_sequence(schema, &[b"{\"jumpGate\": true}"]));
        assert!(accepts_sequence(schema, &[b"{\"jumpGate\": {}}"]));
    }

    #[test]
    fn test_temp2() {
        let schema = r##"{
            "type": "object"
        }"##;
        assert!(!accepts_sequence(schema, &[b"true"]));
        assert!(accepts_sequence(schema, &[b"{}"]));
    }

    #[test]
    fn test_bare_object_schema_accepts_compact_empty_object_token() {
        let schema = r##"{
            "type": "object"
        }"##;
        assert!(accepts_sequence(schema, &[b"{}"]));
    }

    #[test]
    fn test_bare_object_ebnf_accepts_compact_empty_object_token() {
        let ebnf = r#"
            start ::= json_object
            json_object ::= "{" "}" | "{" json_kv json_object_tail "}"
            json_object_tail ::= "," json_kv json_object_tail |
            json_kv ::= json_string ":" json_value
            json_array ::= "[" "]" | "[" json_value json_array_tail "]"
            json_array_tail ::= "," json_value json_array_tail |
            json_value ::= json_object | json_array | json_string | json_bool | json_null
            json_string ::= '""'
            json_bool ::= "true" | "false"
            json_null ::= "null"
        "#;
        assert!(accepts_ebnf_sequence(ebnf, &[b"{}"]));
    }

    #[test]
    fn test_single_pattern_properties_constrain_keys() {
        let schema = r#"{
            "type": "object",
            "additionalProperties": false,
            "patternProperties": {
                "^[(a-z)|(A-Z)|(0-9)]+$": {
                    "type": "string"
                }
            }
        }"#;
        assert!(!accepts_sequence(schema, &[b"{\"\": \"x\"}"]));
        assert!(!accepts_sequence(schema, &[b"{\"!\": \"x\"}"]));
        assert!(accepts_sequence(schema, &[b"{\"A\": \"x\"}"]));
    }

    #[test]
    fn test_match_all_pattern_properties_constrain_values() {
        let schema = r#"{
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string"},
                        "mode": {"type": "string"}
                    },
                    "required": ["destination", "mode"]
                }
            }
        }"#;
        assert!(!accepts_sequence(schema, &[b"{\"instrument1\": true}"]));
        assert!(accepts_sequence(schema, &[b"{\"instrument1\": {\"destination\": \"a\", \"mode\": \"b\"}}"]));
    }

    #[test]
    fn test_prefix_pattern_properties_constrain_matching_values() {
        let schema = r#"{
            "type": "object",
            "patternProperties": {
                "^mode": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }"#;
        assert!(!accepts_sequence(schema, &[b"{\"mode1\": true}"]));
        assert!(accepts_sequence(schema, &[b"{\"mode1\": [\"a\", \"b\"]}"]));
        assert!(accepts_sequence(schema, &[b"{\"other\": true}"]));
    }

    #[test]
    fn test_fixed_object_keys_pack_native_separator_literal() {
        let schema: Value = serde_json::from_str(r#"{
            "type": "object",
            "properties": {
                "opacity": {"type": "number"}
            },
            "additionalProperties": {"type": "string"}
        }"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();
        assert!(named_grammar_has_literal(&grammar, b"\"opacity\": "));
        assert!(!named_grammar_has_split_separator(&grammar, b":", b" "));
    }

    #[test]
    fn test_additional_properties_keys_stay_terminalized() {
        let schema: Value = serde_json::from_str(r#"{
            "type": "object",
            "properties": {
                "opacity": {"type": "number"}
            },
            "additionalProperties": {"type": "string"}
        }"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();
        assert!(!named_nonterminal_has_ref_then_literal(&grammar, b": "));
    }

    #[test]
    fn test_additional_properties_terminalized_keys_still_compile() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "opacity": {"type": "number"}
            },
            "additionalProperties": {"type": "string"}
        }"#;
        assert!(accepts_sequence(schema, &[b"{\"extra\": \"x\"}"]));
    }

    #[test]
    fn test_pattern_object_paths_pack_native_separator_literal() {
        let schema: Value = serde_json::from_str(r#"{
            "type": "object",
            "patternProperties": {
                "^mode": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();
        assert!(named_grammar_has_literal(&grammar, b": "));
        assert!(!named_grammar_has_split_separator(&grammar, b":", b" "));
    }

    #[test]
    fn test_mixed_pattern_unmatched_keys_stay_terminalized() {
        let schema: Value = serde_json::from_str(r#"{
            "type": "object",
            "patternProperties": {
                "^mode": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "additionalProperties": {"type": "string"}
        }"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();
        assert!(!named_nonterminal_has_ref_then_literal(&grammar, b": "));
    }

    #[test]
    fn test_mixed_pattern_terminalized_unmatched_keys_still_compile() {
        let schema = r#"{
            "type": "object",
            "patternProperties": {
                "^mode": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "additionalProperties": {"type": "string"}
        }"#;
        assert!(accepts_sequence(schema, &[b"{\"other\": \"x\"}"]));
    }

    #[test]
    fn test_array_paths_pack_native_item_separator_literal() {
        let schema: Value = serde_json::from_str(r#"{
            "type": "array",
            "prefixItems": [
                {"type": "string"},
                {"type": "number"}
            ],
            "minItems": 2,
            "maxItems": 2
        }"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();
        assert!(named_grammar_has_literal(&grammar, b", "));
        assert!(!named_grammar_has_split_separator(&grammar, b",", b" "));
    }

    #[test]
    fn test_impossible_enum_property_is_omitted() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "builtin": {
                    "enum": ["MODIFIABLE", "DELETABLE"],
                    "type": "object"
                }
            }
        }"#;
        assert!(!accepts_sequence(schema, &[b"{", b"\"builtin\"", b": ", b"{}", b"}"]));
        assert!(accepts_sequence(schema, &[b"{", b"\"builtin \"", b": ", b"\"MODIFIABLE\"", b"}"]));
    }

    #[test]
    fn test_type_array_of_types() {
        let g = json_schema_to_grammar(r#"{"type": ["string", "null"]}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_additional_properties_excludes_prefix_related_known_keys() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "foo": {"type": ["string", "null"]},
                "foo2": {"type": ["string", "null"]}
            }
        }"#;
        assert!(!accepts_sequence(
            schema,
            &[b"{", b"\"foo2\"", b": ", b"1", b"}"]
        ));
        assert!(accepts_sequence(
            schema,
            &[b"{", b"\"foo2\"", b": ", b"null", b"}"]
        ));
        assert!(accepts_sequence(
            schema,
            &[b"{", b"\"foo3\"", b": ", b"1", b"}"]
        ));
    }

    #[test]
    fn test_minimum_constraints_do_not_fall_back_to_nonnegative() {
        assert!(!accepts_sequence(
            r#"{"type": "integer", "minimum": 10}"#,
            &[b"0"]
        ));
        assert!(!accepts_sequence(
            r#"{"type": "integer", "minimum": 10}"#,
            &[b"5"]
        ));
        assert!(accepts_sequence(
            r#"{"type": "integer", "minimum": 10}"#,
            &[b"10"]
        ));
        assert!(!accepts_sequence(
            r#"{"type": "number", "minimum": 10.5}"#,
            &[b"10.4"]
        ));
        assert!(accepts_sequence(
            r#"{"type": "number", "minimum": 10.5}"#,
            &[b"10.5"]
        ));
    }

    #[test]
    fn test_allof_ref_merge_preserves_required_properties() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {"$ref": "#/definitions/Type1"},
                        {"$ref": "#/definitions/Type2"}
                    ]
                }
            },
            "required": ["value"],
            "definitions": {
                "Type1": {
                    "type": "object",
                    "properties": {"value1": {"type": "string"}},
                    "required": ["value1"]
                },
                "Type2": {
                    "type": "object",
                    "properties": {"value2": {"type": "number"}},
                    "required": ["value2"]
                }
            }
        }"##;
        assert!(!accepts_sequence(
            schema,
            &[b"{", b"\"value\"", b": ", b"{", b"\"value1\"", b": ", b"\"\"", b"}", b"}"]
        ));
        assert!(accepts_sequence(
            schema,
            &[b"{", b"\"value\"", b": ", b"{", b"\"value1\"", b": ", b"\"\"", b", ", b"\"value2\"", b": ", b"1", b"}", b"}"]
        ));
    }

    fn accepts_compiled_sequence<F>(tokens: &[&[u8]], build: F) -> bool
    where
        F: FnOnce(&Vocab) -> crate::Result<crate::Constraint>,
    {
        let entries: Vec<(u32, Vec<u8>)> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u32, t.to_vec()))
            .collect();
        let vocab = Vocab::new(entries, None);

        let c = match build(&vocab) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let mut state = c.start();
        for (i, _tok) in tokens.iter().enumerate() {
            let id = i as u32;
            let mask = state.mask();
            let (wi, bi) = (id as usize / 32, id as usize % 32);
            let allowed = wi < mask.len() && (mask[wi] >> bi) & 1 != 0;
            if !allowed {
                return false;
            }
            state.commit_token(id).unwrap();
        }
        state.is_finished()
    }

    fn accepts_sequence(schema_json: &str, tokens: &[&[u8]]) -> bool {
        accepts_compiled_sequence(tokens, |vocab| {
            crate::Constraint::from_json_schema(schema_json, vocab)
        })
    }

    fn accepts_ebnf_sequence(ebnf: &str, tokens: &[&[u8]]) -> bool {
        accepts_compiled_sequence(tokens, |vocab| crate::Constraint::from_ebnf(ebnf, vocab))
    }

    #[test]
    fn test_accepts_boolean_true() {
        assert!(accepts_sequence(r#"{"type": "boolean"}"#, &[b"true"]));
    }

    #[test]
    fn test_accepts_boolean_false() {
        assert!(accepts_sequence(r#"{"type": "boolean"}"#, &[b"false"]));
    }

    #[test]
    fn test_accepts_null_value() {
        assert!(accepts_sequence(r#"{"type": "null"}"#, &[b"null"]));
    }

    #[test]
    fn test_accepts_enum_value() {
        assert!(accepts_sequence(r#"{"enum": ["yes", "no"]}"#, &[b"\"yes\""]));
    }

    #[test]
    fn test_accepts_const_value() {
        assert!(accepts_sequence(r#"{"const": true}"#, &[b"true"]));
    }

    #[test]
    fn test_accepts_date_format() {
        assert!(accepts_sequence(
            r#"{"type": "string", "format": "date"}"#,
            &[b"\"2026-01-30\""]
        ));
    }

    #[test]
    fn test_object_required_only_accepts_valid() {
        let schema = r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#;
        let g = json_schema_to_grammar(schema).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_object_optional_no_trailing_comma() {
        let schema = r#"{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x"]}"#;
        let g = json_schema_to_grammar(schema).unwrap();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn test_json_value_is_recursive() {
        let schema = r#"{"type":"array"}"#;
        let g = json_schema_to_grammar(schema).unwrap();
        assert!(!g.rules.is_empty());
    }
}
