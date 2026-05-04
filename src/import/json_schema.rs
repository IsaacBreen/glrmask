use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::OnceLock;

use serde_json::{Map, Value};
use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr as LexerExpr;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::regex::parse_regex;
use crate::grammar::flat::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, lower, promote_large_literal_alts, expr_to_grammar_expr};

// WARNING: Do NOT break terminals containing repeats of multi-char subexpressions
// into grammar-level repeats of single characters. Doing so creates terminals of
// byte-length 1, which catastrophically bloats the terminal DWA (the parser must
// track every possible single-byte terminal match at every position). Instead,
// keep repeated character patterns fused into chunked multi-char terminals
// (e.g. char{1024}) and use RepeatRange to trigger the direct
// bounded-repeat DFA construction path, which avoids NFA→DFA blowup.

const JSON_VALUE_RULE: &str = "json_value";
const JSON_OBJECT_RULE: &str = "json_object";
const JSON_ARRAY_RULE: &str = "json_array";
const JSON_KV_RULE: &str = "json_kv";
const JSON_STRING_RULE: &str = "json_string";
const JSON_STRING_BODY_RULE: &str = "JSON_STRING_BODY";
const JSON_STRING_MIDDLE_RULE: &str = "JSON_STRING_MIDDLE";
const JSON_STRING_MIDDLE_END_RULE: &str = "JSON_STRING_MIDDLE_END";
const JSON_STRING_CHAR_RULE: &str = "JSON_STRING_CHAR";
const JSON_INTEGER_RULE: &str = "JSON_INTEGER";
const JSON_NUMBER_RULE: &str = "JSON_NUMBER";
const JSON_NONNEG_INTEGER_RULE: &str = "JSON_NONNEG_INTEGER";
const JSON_NONNEG_NUMBER_RULE: &str = "JSON_NONNEG_NUMBER";
const JSON_BOOL_RULE: &str = "JSON_BOOL";
const JSON_NULL_RULE: &str = "JSON_NULL";
const JSON_KEY_COLON_RULE: &str = "json_key_colon";
const JSON_KEY_COLON_BODY_RULE: &str = "JSON_KEY_COLON_BODY";
const JSON_STRING_REPEAT_CHUNK_DEFAULT: usize = 256;

fn json_string_repeat_chunk() -> usize {
    std::env::var("GLRMASK_STRING_REPEAT_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(JSON_STRING_REPEAT_CHUNK_DEFAULT)
}

const JSON_STRING_BODY_REGEX: &str =
    r#"([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*""#;
/// Body chars only, no surrounding quotes.
const JSON_STRING_BODY_ONLY_REGEX: &str =
    r#"([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"#;
/// Full JSON string including both opening and closing quotes.
const JSON_STRING_FULL_REGEX: &str =
    r#""([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*""#;
/// Opening quote + body chars, no closing quote.
const JSON_STRING_OPEN_BODY_REGEX: &str =
    r#""([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"#;
const JSON_KEY_COLON_BODY_REGEX: &str =
    r#"([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*": "#;
/// Full key+colon including opening quote: `"key": `.
const JSON_KEY_COLON_FULL_REGEX: &str =
    r#""([^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*": "#;
const JSON_STRING_CHAR_PATTERN: &str = r#"[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}"#;
const JSON_NUMBER_NONINTEGER_REGEX: &str =
    r#"-?(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)"#;
const JSON_NONNEG_NUMBER_NONINTEGER_REGEX: &str =
    r#"(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)"#;
const JSON_DIRECT_UTF8_PATTERN: &str =
    r#"(?:[\xC2-\xDF][\x80-\xBF]|[\xE0][\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC][\x80-\xBF][\x80-\xBF]|[\xED][\x80-\x9F][\x80-\xBF]|[\xEE-\xEF][\x80-\xBF][\x80-\xBF]|[\xF0][\x90-\xBF][\x80-\xBF][\x80-\xBF]|[\xF1-\xF3][\x80-\xBF][\x80-\xBF][\x80-\xBF]|[\xF4][\x80-\x8F][\x80-\xBF][\x80-\xBF])"#;
const JSON_ITEM_SEPARATOR: &[u8] = b", ";
const JSON_KEY_SEPARATOR: &[u8] = b": ";
const CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_ALTS: usize = 128;
const CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_TOTAL_BYTES: usize = 64 * 1024;
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
const EXACT_CLOSED_OBJECT_UNION_MAX_VARIANTS: usize = 8;
const EXACT_CLOSED_OBJECT_UNION_MAX_KEYS: usize = 16;
const EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS: usize = 16;
const EXACT_CLOSED_OBJECT_UNION_MAX_STATES: usize = 128;
const FACTORED_OPEN_OBJECT_MAX_KEYS: usize = 200;

fn env_usize_with_default(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn cached_env_usize(slot: &'static OnceLock<usize>, name: &'static str, default: usize) -> usize {
    *slot.get_or_init(|| env_usize_with_default(name, default))
}

fn closed_required_object_fused_literal_max_alts() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_ALTS",
        CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_ALTS,
    )
}

fn closed_required_object_fused_literal_max_total_bytes() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_TOTAL_BYTES",
        CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_TOTAL_BYTES,
    )
}

fn exact_closed_object_union_max_variants() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_EXACT_CLOSED_OBJECT_UNION_MAX_VARIANTS",
        EXACT_CLOSED_OBJECT_UNION_MAX_VARIANTS,
    )
}

fn exact_closed_object_union_max_keys() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_EXACT_CLOSED_OBJECT_UNION_MAX_KEYS",
        EXACT_CLOSED_OBJECT_UNION_MAX_KEYS,
    )
}

fn exact_closed_object_single_max_keys() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS",
        EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS,
    )
}

fn exact_closed_object_union_max_states() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_EXACT_CLOSED_OBJECT_UNION_MAX_STATES",
        EXACT_CLOSED_OBJECT_UNION_MAX_STATES,
    )
}

fn factored_open_object_max_keys() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    cached_env_usize(
        &VALUE,
        "GLRMASK_FACTORED_OPEN_OBJECT_MAX_KEYS",
        FACTORED_OPEN_OBJECT_MAX_KEYS,
    )
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum JsonSchemaDraft {
    Draft4,
    Draft6,
    Draft7,
    Draft201909,
    Draft202012,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralBranchMode {
    AnyOf,
    OneOf,
}

#[derive(Debug, Clone)]
struct OrderedClosedObjectSchemaItem {
    key: String,
    value_schema: Value,
    required: bool,
}

#[derive(Debug, Clone)]
struct OrderedClosedObjectSchemaVariant {
    items: Vec<OrderedClosedObjectSchemaItem>,
}

#[derive(Debug, Clone)]
struct OrderedClosedObjectItem {
    key: String,
    value_expr: GrammarExpr,
    required: bool,
}

#[derive(Debug, Clone)]
struct OrderedClosedObjectVariant {
    items: Vec<OrderedClosedObjectItem>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct OrderedSubsetCursor {
    variant_idx: u16,
    cursor: u16,
}

impl OrderedClosedObjectVariant {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn advance_cursor(&self, cursor: usize, key: &str) -> Option<usize> {
        let idx = self.items[cursor..]
            .iter()
            .position(|item| item.key == key)
            .map(|offset| cursor + offset)?;
        if self.items[cursor..idx].iter().any(|item| item.required) {
            return None;
        }
        Some(idx + 1)
    }

    fn close_allowed(&self, cursor: usize) -> bool {
        !self.items[cursor..].iter().any(|item| item.required)
    }

    fn legal_next_keys(&self, cursor: usize) -> Vec<&str> {
        let mut keys = Vec::new();
        for item in &self.items[cursor..] {
            keys.push(item.key.as_str());
            if item.required {
                break;
            }
        }
        keys
    }

    fn value_expr_for_key(&self, key: &str) -> Option<GrammarExpr> {
        self.items
            .iter()
            .find(|item| item.key == key)
            .map(|item| item.value_expr.clone())
    }
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

fn normalize_numeric_bounds(schema: &Map<String, Value>) -> (Option<f64>, bool, Option<f64>, bool) {
    let minimum = schema.get("minimum").and_then(Value::as_f64);
    let maximum = schema.get("maximum").and_then(Value::as_f64);
    let exclusive_minimum = schema.get("exclusiveMinimum");
    let exclusive_maximum = schema.get("exclusiveMaximum");

    let mut left = minimum;
    let mut left_inclusive = true;
    match exclusive_minimum {
        Some(Value::Bool(true)) => {
            left = minimum;
            left_inclusive = false;
        }
        Some(value) => {
            if let Some(exclusive_minimum) = value.as_f64() {
                if left.is_none() || exclusive_minimum >= left.unwrap() {
                    left = Some(exclusive_minimum);
                    left_inclusive = false;
                }
            }
        }
        None => {}
    }

    let mut right = maximum;
    let mut right_inclusive = true;
    match exclusive_maximum {
        Some(Value::Bool(true)) => {
            right = maximum;
            right_inclusive = false;
        }
        Some(value) => {
            if let Some(exclusive_maximum) = value.as_f64() {
                if right.is_none() || exclusive_maximum <= right.unwrap() {
                    right = Some(exclusive_maximum);
                    right_inclusive = false;
                }
            }
        }
        None => {}
    }

    (left, left_inclusive, right, right_inclusive)
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

fn dedup_literal_alternatives(mut alts: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    if alts.len() > 1 {
        alts.sort();
        alts.dedup();
    }
    alts
}

fn finite_literal_alternatives(
    expr: &GrammarExpr,
    max_alts: usize,
) -> Option<Vec<Vec<u8>>> {
    let alts = match expr {
        GrammarExpr::Literal(bytes) => vec![bytes.clone()],
        GrammarExpr::Sequence(parts) => {
            let mut acc = vec![Vec::new()];
            for part in parts {
                let part_alts = finite_literal_alternatives(part, max_alts)?;
                let max_product = acc.len().checked_mul(part_alts.len())?;
                if max_product > max_alts {
                    return None;
                }

                let mut next = Vec::with_capacity(max_product);
                for prefix in &acc {
                    for suffix in &part_alts {
                        let mut bytes = Vec::with_capacity(prefix.len() + suffix.len());
                        bytes.extend_from_slice(prefix);
                        bytes.extend_from_slice(suffix);
                        next.push(bytes);
                    }
                }
                acc = dedup_literal_alternatives(next);
                if acc.len() > max_alts {
                    return None;
                }
            }
            acc
        }
        GrammarExpr::Choice(options) => {
            let mut out = Vec::new();
            for option in options {
                out.extend(finite_literal_alternatives(option, max_alts)?);
                out = dedup_literal_alternatives(out);
                if out.len() > max_alts {
                    return None;
                }
            }
            out
        }
        GrammarExpr::Optional(inner) => {
            let mut out = vec![Vec::new()];
            out.extend(finite_literal_alternatives(inner, max_alts)?);
            dedup_literal_alternatives(out)
        }
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Exclude { .. }
        | GrammarExpr::Repeat(_)
        | GrammarExpr::RepeatOne(_)
        | GrammarExpr::RepeatRange { .. }
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::AnyByte
        | GrammarExpr::Intersect { .. }
        | GrammarExpr::SeparatedSequence { .. } => return None,
    };

    let total_bytes = alts.iter().map(Vec::len).sum::<usize>();
    if alts.len() > max_alts || total_bytes > closed_required_object_fused_literal_max_total_bytes() {
        return None;
    }

    Some(alts)
}

fn maybe_fuse_finite_literal_expr(expr: GrammarExpr, context: &str) -> GrammarExpr {
    let Some(alts) = finite_literal_alternatives(&expr, closed_required_object_fused_literal_max_alts()) else {
        return expr;
    };
    if alts.len() < 2 {
        return expr;
    }

    if std::env::var("GLRMASK_PROFILE_OBJECT_FUSION").is_ok() {
        let total_bytes = alts.iter().map(Vec::len).sum::<usize>();
        let max_bytes = alts.iter().map(Vec::len).max().unwrap_or(0);
        eprintln!(
            "[glrmask/profile][object_fusion] context={} alts={} total_bytes={} max_bytes={}",
            context,
            alts.len(),
            total_bytes,
            max_bytes,
        );
    }

    choice_or_single(
        alts.into_iter()
            .map(GrammarExpr::Literal)
            .collect(),
    )
}

fn try_append_suffix_to_trailing_literal(expr: GrammarExpr, suffix: &[u8]) -> Option<GrammarExpr> {
    match expr {
        GrammarExpr::Literal(mut bytes) => {
            if matches!(bytes.last(), Some(b'}') | Some(b']')) {
                bytes.extend_from_slice(suffix);
                Some(GrammarExpr::Literal(bytes))
            } else {
                None
            }
        }
        GrammarExpr::Sequence(mut parts) => {
            let last = parts.pop()?;
            let new_last = try_append_suffix_to_trailing_literal(last, suffix)?;
            parts.push(new_last);
            Some(sequence_or_single(parts))
        }
        GrammarExpr::Choice(options) => {
            let mut fused = Vec::with_capacity(options.len());
            for option in options {
                fused.push(try_append_suffix_to_trailing_literal(option, suffix)?);
            }
            Some(choice_or_single(fused))
        }
        _ => None,
    }
}

fn try_take_leading_container_delim(expr: GrammarExpr) -> Option<(u8, GrammarExpr)> {
    match expr {
        GrammarExpr::Literal(mut bytes) => {
            let first = *bytes.first()?;
            if !matches!(first, b'{' | b'[') {
                return None;
            }
            bytes.remove(0);
            let rest = if bytes.is_empty() {
                empty_expr()
            } else {
                literal_expr(&bytes)
            };
            Some((first, rest))
        }
        GrammarExpr::Sequence(mut parts) => {
            if parts.is_empty() {
                return None;
            }
            let first_part = parts.remove(0);
            let (first, rest_first) = try_take_leading_container_delim(first_part)?;
            let mut rebuilt = Vec::new();
            match rest_first {
                GrammarExpr::Sequence(inner) => rebuilt.extend(inner),
                GrammarExpr::Literal(bytes) if bytes.is_empty() => {}
                other => rebuilt.push(other),
            }
            rebuilt.extend(parts);
            Some((first, sequence_or_single(rebuilt)))
        }
        GrammarExpr::Choice(options) => {
            let mut stripped = Vec::with_capacity(options.len());
            let mut first_byte = None;
            for option in options {
                let (candidate_first, rest) = try_take_leading_container_delim(option)?;
                if let Some(expected_first) = first_byte {
                    if expected_first != candidate_first {
                        return None;
                    }
                } else {
                    first_byte = Some(candidate_first);
                }
                stripped.push(rest);
            }
            Some((first_byte?, choice_or_single(stripped)))
        }
        _ => None,
    }
}

fn json_format_pattern(format_name: &str) -> Option<&'static str> {
    Some(match format_name {
        "time" => {
            r#"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"#
        }
        "duration" => {
            r#"P(?:(?:(?:[0-9]+Y(?:[0-9]+M(?:[0-9]+D)?)?)|(?:[0-9]+M(?:[0-9]+D)?)|(?:[0-9]+D))(?:T(?:(?:[0-9]+H(?:[0-9]+M(?:[0-9]+S)?)?)|(?:[0-9]+M(?:[0-9]+S)?)|(?:[0-9]+S)))?)|(?:T(?:(?:[0-9]+H(?:[0-9]+M(?:[0-9]+S)?)?)|(?:[0-9]+M(?:[0-9]+S)?)|(?:[0-9]+S)))|(?:[0-9]+W)"#
        }
        "email" => {
            r#"(?:(?:[a-zA-Z0-9!#$%&'*+\-/=?\^_`{|}~]+(?:\.[a-zA-Z0-9!#$%&'*+\-/=?\^_`{|}~]+)*))@((?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)(?:\.(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?))*)|\[(?:((([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))\.){3}(([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9]))))\]"#
        }
        "hostname" => {
            r#"[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*"#
        }
        "ipv4" => {
            r#"((([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))\.){3}(([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))"#
        }
        "ipv6" => {
            r#"(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4})|(?:::(?:[0-9a-fA-F]{1,4}:){0,5}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:(?:[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,4}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,1}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,3}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,2}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,2}(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,3}[0-9a-fA-F]{1,4})?::[0-9a-fA-F]{1,4}:(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,4}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}))|(?:((?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4})?::)"#
        }
        "uuid" => r#"(?:[0-9a-fA-F]{8})-(?:[0-9a-fA-F]{4})-(?:[0-9a-fA-F]{4})-(?:[0-9a-fA-F]{2})(?:[0-9a-fA-F]{2})-(?:[0-9a-fA-F]{12})"#,
        "uri" => {
            r#"(?:[a-zA-Z][a-zA-Z0-9+\-.]*):(?://(?:(?:(?:[a-zA-Z0-9\-._~!$&'()*+,;=:]|%[0-9a-fA-F]{2})*)@)?(?:\[(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}|:(?::[0-9a-fA-F]{1,4}){1,7}|::|v[0-9a-fA-F]+\.[a-zA-Z0-9\-._~!$&'()*+,;=:]+)\]|(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])|(?:[a-zA-Z0-9\-._~!$&'()*+,;=]|%[0-9a-fA-F]{2})*)(?::(?:[0-9]*))?(?:(?:/(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)|(?:/(?:(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})+(?:/(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)?)|(?:(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})+(?:/(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)|(?:))(?:\?(?:(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]|%[0-9a-fA-F]{2})*))?(?:\#(?:(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]|%[0-9a-fA-F]{2})*))?"#
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

fn strip_branch_outer_anchors(branch: &str) -> (bool, bool, &str) {
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
        (false, false) => match max_tail {
            Some(n) => {
                let constrained_tails = (0..=n)
                    .map(|left_budget| {
                        let right_budget = n.saturating_sub(left_budget);
                        format!(
                            r#"(?:{}){{0,{}}}(?:{})(?:{}){{0,{}}}"#,
                            JSON_STRING_CHAR_PATTERN,
                            left_budget,
                            inner,
                            JSON_STRING_CHAR_PATTERN,
                            right_budget,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("|");
                format!("(?:{})", constrained_tails)
            }
            None => format!("{}(?:{}){}", string_tail, inner, string_tail),
        },
    }
}

fn json_search_pattern(pattern: &str) -> String {
    json_search_pattern_impl(pattern, None)
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
        Expr::Exclude { expr, .. } => regex_byte_length_bounds(expr),
        Expr::Intersect { expr, intersect } => {
            let (left_min, left_max) = regex_byte_length_bounds(expr);
            let (right_min, right_max) = regex_byte_length_bounds(intersect);
            let min_total = left_min.max(right_min);
            let max_total = match (left_max, right_max) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
            };
            (min_total, max_total)
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

/// Compute the minimum number of logical characters a regex pattern can match.
///
/// Operates on the pattern string directly rather than the byte-level AST,
/// so it correctly handles multi-byte Unicode characters (each char-class
/// atom or literal counts as one character regardless of UTF-8 byte width).
///
/// Returns `None` when the pattern contains any construct the walker does
/// not recognise (backreferences, Unicode property escapes, flag groups, …).
/// The caller should fall back to a conservative tail budget in that case.
fn pattern_min_char_count(pattern: &str) -> Option<usize> {
    let branches = split_top_level_regex_branches(pattern);
    let mut overall_min = usize::MAX;
    for b in &branches {
        overall_min = overall_min.min(pattern_branch_min_chars(b.as_bytes())?);
    }
    Some(if overall_min == usize::MAX { 0 } else { overall_min })
}

fn pattern_branch_min_chars(bytes: &[u8]) -> Option<usize> {
    let mut i = 0;
    let mut total = 0usize;
    while i < bytes.len() {
        let (atom_min, next) = pattern_atom_min_chars(bytes, i)?;
        i = next;
        let (rep_min, next) = pattern_quantifier_min(bytes, i);
        i = next;
        total = total.saturating_add(atom_min.saturating_mul(rep_min));
    }
    Some(total)
}

/// Read one regex atom starting at position `i`.
/// Returns `None` for unrecognised constructs.
fn pattern_atom_min_chars(bytes: &[u8], i: usize) -> Option<(usize, usize)> {
    if i >= bytes.len() {
        return Some((0, i));
    }
    match bytes[i] {
        b'^' | b'$' => Some((0, i + 1)),
        b'.' => Some((1, i + 1)),
        b'|' => Some((0, i + 1)), // should not appear at branch level; advance to prevent loop
        b'[' => {
            let end = pattern_skip_char_class(bytes, i);
            Some((1, end))
        }
        b'(' => pattern_group_min_chars(bytes, i),
        b'\\' if i + 1 < bytes.len() => pattern_escape_min_chars(bytes, i),
        b'\\' => None, // trailing backslash — malformed
        _ if bytes[i] < 0x80 => Some((1, i + 1)), // ASCII literal
        _ => {
            // Multi-byte UTF-8 literal.
            let byte = bytes[i];
            let char_len = if byte < 0xE0 { 2 } else if byte < 0xF0 { 3 } else { 4 };
            Some((1, (i + char_len).min(bytes.len())))
        }
    }
}

/// Handle a backslash escape starting at position `i`.
/// Returns `None` for unrecognised / unsupported escape sequences
/// (backreferences, Unicode properties, control escapes, etc.).
fn pattern_escape_min_chars(bytes: &[u8], i: usize) -> Option<(usize, usize)> {
    let escaped = bytes[i + 1];
    match escaped {
        // Zero-width assertions
        b'b' | b'B' => Some((0, i + 2)),
        // Character class shorthands — each matches 1 character
        b'd' | b'D' | b'w' | b'W' | b's' | b'S' => Some((1, i + 2)),
        // Named control-character escapes — each matches 1 character
        b't' | b'n' | b'r' | b'f' | b'v' | b'0' => Some((1, i + 2)),
        // Hex escape \xHH — 1 character
        b'x' if i + 3 < bytes.len()
            && bytes[i + 2].is_ascii_hexdigit()
            && bytes[i + 3].is_ascii_hexdigit() =>
        {
            Some((1, i + 4))
        }
        // Unicode escape \uHHHH — 1 character
        b'u' if i + 5 < bytes.len()
            && bytes[i + 2].is_ascii_hexdigit()
            && bytes[i + 3].is_ascii_hexdigit()
            && bytes[i + 4].is_ascii_hexdigit()
            && bytes[i + 5].is_ascii_hexdigit() =>
        {
            Some((1, i + 6))
        }
        // Backreferences \1..\9 — length depends on captured group
        b'1'..=b'9' => None,
        // Unicode property escapes \p{..}, \P{..}
        b'p' | b'P' => None,
        // Named backreference \k<name>
        b'k' => None,
        // Control escape \cX
        b'c' => None,
        // Escaped non-alphanumeric ASCII: literal metachar (1 character)
        _ if !escaped.is_ascii_alphanumeric() => Some((1, i + 2)),
        // Any other alphabetic escape — unknown, bail out
        _ => None,
    }
}

/// Skip from '[' to the position after the closing ']'.
fn pattern_skip_char_class(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    if i < bytes.len() && bytes[i] == b'^' {
        i += 1;
    }
    // First char after [^ can be ']' as a literal member.
    if i < bytes.len() && bytes[i] == b']' {
        i += 1;
    }
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
        } else if bytes[i] == b']' {
            return i + 1;
        } else {
            i += 1;
        }
    }
    i
}

/// Parse a parenthesized group and return (min_chars, position_after_close_paren).
/// Returns `None` for unrecognised group syntax (named groups, flag groups, …).
fn pattern_group_min_chars(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut i = start + 1; // skip '('

    if i < bytes.len() && bytes[i] == b'?' {
        i += 1;
        if i >= bytes.len() {
            return None;
        }
        match bytes[i] {
            b':' => {
                // Non-capturing group (?:...) — parse contents normally.
                i += 1;
            }
            b'=' | b'!' => {
                // Lookahead (?=...) or (?!...) — zero-width.
                i += 1;
                let end = pattern_skip_group_close(bytes, i);
                return Some((0, end));
            }
            b'<' if i + 1 < bytes.len() && matches!(bytes[i + 1], b'=' | b'!') => {
                // Lookbehind (?<=...) or (?<!...) — zero-width.
                i += 2;
                let end = pattern_skip_group_close(bytes, i);
                return Some((0, end));
            }
            _ => {
                // Named group (?<name>...), flag group (?i...), etc. — unsupported.
                return None;
            }
        }
    }
    // Regular capturing group (...) or non-capturing (?:...) — parse branches.

    let mut branch_total = 0usize;
    let mut min_across = usize::MAX;

    while i < bytes.len() && bytes[i] != b')' {
        if bytes[i] == b'|' {
            min_across = min_across.min(branch_total);
            branch_total = 0;
            i += 1;
            continue;
        }
        let (atom_min, next) = pattern_atom_min_chars(bytes, i)?;
        i = next;
        let (rep_min, next) = pattern_quantifier_min(bytes, i);
        i = next;
        branch_total = branch_total.saturating_add(atom_min.saturating_mul(rep_min));
    }

    min_across = min_across.min(branch_total);
    if min_across == usize::MAX {
        min_across = 0;
    }
    if i < bytes.len() && bytes[i] == b')' {
        Some((min_across, i + 1))
    } else {
        None // unterminated group
    }
}

/// Skip forward to the position after the matching ')'.
fn pattern_skip_group_close(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    let mut depth = 1u32;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth -= 1;
                i += 1;
            }
            b'\\' if i + 1 < bytes.len() => {
                i += 2;
            }
            b'[' => {
                i = pattern_skip_char_class(bytes, i);
            }
            _ => {
                i += 1;
            }
        }
    }
    i
}

/// Parse an optional quantifier and return (min_repetitions, next_position).
/// Returns (1, same_pos) when no quantifier is present.
fn pattern_quantifier_min(bytes: &[u8], i: usize) -> (usize, usize) {
    if i >= bytes.len() {
        return (1, i);
    }
    let lazy_skip = |pos: usize| -> usize {
        if pos < bytes.len() && bytes[pos] == b'?' {
            pos + 1
        } else {
            pos
        }
    };
    match bytes[i] {
        b'*' => (0, lazy_skip(i + 1)),
        b'+' => (1, lazy_skip(i + 1)),
        b'?' => (0, lazy_skip(i + 1)),
        b'{' => {
            let mut j = i + 1;
            let mut n = 0usize;
            let mut has_digit = false;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                n = n.saturating_mul(10).saturating_add((bytes[j] - b'0') as usize);
                has_digit = true;
                j += 1;
            }
            if !has_digit {
                return (1, i); // not a quantifier, literal '{'
            }
            if j < bytes.len() && bytes[j] == b'}' {
                (n, lazy_skip(j + 1)) // {n}
            } else if j < bytes.len() && bytes[j] == b',' {
                j += 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'}' {
                    (n, lazy_skip(j + 1)) // {n,} or {n,m}
                } else {
                    (1, i) // malformed
                }
            } else {
                (1, i) // not a quantifier
            }
        }
        _ => (1, i),
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
/// sequences, direct multi-byte UTF-8 patterns, and `\\uXXXX` encodings
/// (as an over-approximation — all `\\uXXXX` values are accepted regardless
/// of whether the encoded codepoint is in the excluded set).
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

    // JSON \uXXXX escape sequences (over-approximation: accepts all
    // codepoints, not just those outside the excluded set).
    parts.push(String::from(r#"\x5Cu[0-9A-Fa-f]{4}"#));

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

fn env_flag(name: &str) -> bool {
    env_flag_default(name, false)
}

fn split_open_quote() -> bool {
    // Default to true: opening quote is split unless explicitly disabled
    env_flag_default("GLRMASK_SPLIT_OPEN_QUOTE", true)
}

fn split_close_quote() -> bool {
    env_flag("GLRMASK_SPLIT_CLOSE_QUOTE")
}

fn split_colon_space() -> bool {
    // Default to true: colon-space is split unless explicitly disabled
    env_flag_default("GLRMASK_SPLIT_COLON_SPACE", true)
}

fn split_colon_from_space() -> bool {
    env_flag("GLRMASK_SPLIT_COLON_FROM_SPACE")
}

fn split_item_separator() -> bool {
    // Default to fused JSON_ITEM_SEPARATOR literal unless explicitly enabled.
    env_flag("GLRMASK_SPLIT_ITEM_SEPARATOR")
}

// ---------------------------------------------------------------------------
// Hierarchical string / key-colon construction
// ---------------------------------------------------------------------------
//
// All JSON string values and key-colon terminals are built through a
// three-layer hierarchy:
//
//   Layer 1 (body):  The content regex/literal/DFA with NO quotes or
//                    colon-space.  Callers construct this themselves.
//
//   Layer 2 (quotes): Fuses non-split quotes into the body and emits
//                     split quotes as separate literal terminals.
//                     Used for string VALUES (`"body"`).
//
//   Layer 3 (key-colon): Extends Layer 2 by also handling the trailing
//                        `": "` delimiter.  Used for object KEYS
//                        (`"body": `).
//
// Each layer is solely responsible for its own split flags.  Callers
// never need to inspect split flags directly — they just produce a body
// and hand it to the appropriate wrapper.
//
// Two parallel APIs exist for each layer:
//  - **Regex patterns** (`*_body_regex`): return a regex string whose
//    accepted language includes exactly the fused delimiters.  The caller
//    converts this to a `regex_expr()` or builds a DFA from it.
//  - **Terminal wrappers** (`wrap_*_terminal`): take an already-built
//    `GrammarExpr` body terminal and surround it with the split-off
//    literal terminals.
//
// Convenience combinators (`wrap_*_regex`) compose both steps.
// ---------------------------------------------------------------------------

/// Build the regex pattern for the body terminal of a JSON string **value**.
///
/// Fuses non-split quotes into the pattern:
///  - `split_open=false` → pattern starts with `"`
///  - `split_close=false` → pattern ends with `"`
///
/// The caller wraps the result with `regex_expr()` and passes it to
/// `wrap_string_value_terminal()`.
fn string_value_body_regex(inner: &str) -> String {
    let open = split_open_quote();
    let close = split_close_quote();
    match (open, close) {
        (false, false) => format!(r#""(?:{})""#, inner),
        (false, true)  => format!(r#""(?:{})"#, inner),
        (true, false)  => format!(r#"(?:{})""#, inner),
        (true, true)   => format!(r#"(?:{})"#, inner),
    }
}

/// Build the regex pattern for the body terminal of a JSON object **key-colon**
/// (`"key": `).
///
/// Fuses non-split quotes AND colon-space into the pattern:
///  - `split_open=false` → pattern starts with `"`
///  - `split_close=false` → pattern includes close `"`
///  - `split_colon=false` → pattern includes `": "`
fn key_colon_body_regex(inner: &str) -> String {
    let open = split_open_quote();
    let close = split_close_quote();
    let colon = split_colon_space();
    match (open, close, colon) {
        (false, false, false) => format!(r#""(?:{})": "#, inner),
        (false, false, true)  => format!(r#""(?:{})""#, inner),
        (false, true, _)      => format!(r#""(?:{})"#, inner),
        (true, false, false)  => format!(r#"(?:{})": "#, inner),
        (true, false, true)   => format!(r#"(?:{})""#, inner),
        (true, true, _)       => format!(r#"(?:{})"#, inner),
    }
}

/// Build literal bytes for the body terminal of a JSON object **key-colon**.
///
/// Fuses non-split quotes and colon-space into the byte sequence.
fn key_colon_literal_body_bytes(text: &str) -> Vec<u8> {
    let full = json_string_literal_bytes(text); // b'"text"'
    let body_only = &full[1..full.len() - 1];
    let open = split_open_quote();
    let close = split_close_quote();
    let colon = split_colon_space();
    let mut bytes = Vec::new();
    if !open { bytes.push(b'"'); }
    bytes.extend_from_slice(body_only);
    match (close, colon) {
        (false, false) => bytes.extend_from_slice(b"\": "),
        (false, true)  => bytes.push(b'"'),
        (true, _)      => {} // close quote and colon handled by wrapper
    }
    bytes
}

/// Build the colon-space suffix expression for keys.
fn key_colon_suffix_expr() -> GrammarExpr {
    if split_colon_from_space() {
        sequence_or_single(vec![literal_expr(b":"), literal_expr(b" ")])
    } else {
        literal_expr(b": ")
    }
}

/// Wrap a body terminal expression as a JSON string **value**.
///
/// Adds split-off quote literals around the body terminal.
/// The body terminal must already include fused (non-split) quotes.
fn wrap_string_value_terminal(body: GrammarExpr) -> GrammarExpr {
    let open = split_open_quote();
    let close = split_close_quote();
    let mut parts = Vec::new();
    if open { parts.push(literal_expr(b"\"")); }
    parts.push(body);
    if close { parts.push(literal_expr(b"\"")); }
    sequence_or_single(parts)
}

/// Wrap a body terminal expression as a JSON object **key-colon**.
///
/// Adds split-off quote literals and colon-space around the body terminal.
/// The body terminal must already include fused (non-split) quotes and colon.
fn wrap_key_colon_terminal(body: GrammarExpr) -> GrammarExpr {
    let open = split_open_quote();
    let close = split_close_quote();
    let colon = split_colon_space();
    let mut parts = Vec::new();
    if open { parts.push(literal_expr(b"\"")); }
    parts.push(body);
    if close && colon {
        parts.push(literal_expr(b"\""));
        parts.push(key_colon_suffix_expr());
    } else if close {
        parts.push(literal_expr(b"\": "));
    } else if colon {
        parts.push(key_colon_suffix_expr());
    }
    sequence_or_single(parts)
}

// ---------------------------------------------------------------------------
// Convenience combinators
// ---------------------------------------------------------------------------

/// Wrap a body regex as a JSON key-colon expression.
/// Shorthand for `wrap_key_colon_terminal(regex_expr(key_colon_body_regex(inner)))`.
fn wrap_key_colon_regex(inner_regex: &str) -> GrammarExpr {
    wrap_key_colon_terminal(regex_expr(key_colon_body_regex(inner_regex)))
}

fn parsed_regex_expr(pattern: &str, utf8: bool) -> GrammarExpr {
    expr_to_grammar_expr(&parse_regex(pattern, utf8))
}

fn no_additional_properties() -> bool {
    env_flag("GLRMASK_NO_ADDITIONAL_PROPERTIES")
}

fn additional_properties_default_false() -> bool {
    env_flag("GLRMASK_AP_DEFAULT_FALSE")
}

fn ap_key_any_string() -> bool {
    env_flag("GLRMASK_AP_KEY_ANY_STRING") || env_flag("GLRMASK_ADDPROP_NO_EXCLUSIONS")
}

fn shared_ap_key_exclusions_enabled() -> bool {
    std::env::var("GLRMASK_AP_SHARED_EXCLUSIONS")
        .map(|v| {
            let n = v.trim().to_ascii_lowercase();
            !matches!(n.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}

fn decode_local_ref_token(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn find_local_ref_target<'v>(node: &'v Value, ref_value: &str) -> Option<&'v Value> {
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
                if let Some(target) = find_local_ref_target(value, ref_value) {
                    return Some(target);
                }
            }
            None
        }
        Value::Array(items) => items
            .iter()
            .find_map(|value| find_local_ref_target(value, ref_value)),
        _ => None,
    }
}

fn resolve_shared_ap_ref_target<'a>(root: &'a Value, ref_value: &str) -> Option<&'a Value> {
    if !ref_value.starts_with('#') {
        return None;
    }

    if ref_value == "#" {
        return Some(root);
    }

    if !ref_value.starts_with("#/") {
        return find_local_ref_target(root, ref_value);
    }

    let mut current = root;
    for token in ref_value[2..].split('/') {
        let key = decode_local_ref_token(token);
        current = current.get(&key)?;
    }
    Some(current)
}

fn collect_shared_ap_literal_keys(root: &Value) -> BTreeSet<String> {
    let mut collected = BTreeSet::new();
    let mut queue = VecDeque::from([root]);
    let mut visited = BTreeSet::new();

    while let Some(node) = queue.pop_front() {
        let node_id = node as *const Value as usize;
        if !visited.insert(node_id) {
            continue;
        }

        let Some(object) = node.as_object() else {
            continue;
        };

        let additional_properties = object.get("additionalProperties");
        let allows_additional_properties = !matches!(additional_properties, Some(Value::Bool(false)));
        if allows_additional_properties {
            if let Some(properties) = object.get("properties").and_then(Value::as_object) {
                collected.extend(properties.keys().cloned());
            }
            if let Some(required) = object.get("required").and_then(Value::as_array) {
                collected.extend(required.iter().filter_map(Value::as_str).map(String::from));
            }
        }

        if let Some(properties) = object.get("properties").and_then(Value::as_object) {
            queue.extend(properties.values());
        }
        if let Some(pattern_properties) = object.get("patternProperties").and_then(Value::as_object) {
            queue.extend(pattern_properties.values());
        }

        if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
            if let Some(target) = resolve_shared_ap_ref_target(root, reference) {
                queue.push_back(target);
            }
        }

        for keyword in [
            "additionalProperties",
            "propertyNames",
            "items",
            "additionalItems",
            "contains",
            "unevaluatedProperties",
            "if",
            "then",
            "else",
            "not",
        ] {
            if let Some(value) = object.get(keyword) {
                queue.push_back(value);
            }
        }

        for keyword in ["prefixItems", "allOf", "anyOf", "oneOf"] {
            if let Some(values) = object.get(keyword).and_then(Value::as_array) {
                queue.extend(values);
            }
        }

        for keyword in ["$defs", "definitions", "dependentSchemas"] {
            if let Some(values) = object.get(keyword).and_then(Value::as_object) {
                queue.extend(values.values());
            }
        }
    }

    collected
}

const SHARED_AP_MAX_ALLOW_BACK_KEYS: usize = 32;

fn max_string_length_cap() -> Option<usize> {
    std::env::var("GLRMASK_MAX_STRING_LENGTH_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

fn use_structured_uri() -> bool {
    env_flag_default("GLRMASK_STRUCT_URI_FORMAT", true)
}

fn env_flag_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|v| {
            let n = v.trim().to_ascii_lowercase();
            !matches!(n.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}

fn uri_rule_should_be_terminal(name: &str) -> Option<bool> {
    match name {
        "uri_scheme" => Some(env_flag_default("GLRMASK_URI_SCHEME_TERMINAL", false)),
        "uri_alpha_char" => Some(env_flag_default("GLRMASK_URI_ALPHA_CHAR_TERMINAL", true)),
        "uri_scheme_char" => Some(env_flag_default("GLRMASK_URI_SCHEME_CHAR_TERMINAL", true)),
        "uri_reg_name_char" => Some(env_flag_default("GLRMASK_URI_REG_NAME_CHAR_TERMINAL", true)),
        "uri_pchar_char" => Some(env_flag_default("GLRMASK_URI_PCHAR_CHAR_TERMINAL", false)),
        "uri_query_frag_char" => {
            Some(env_flag_default("GLRMASK_URI_QUERY_FRAG_CHAR_TERMINAL", true))
        }
        "uri_pchar" => Some(env_flag_default("GLRMASK_URI_PCHAR_TERMINAL", false)),
        "uri_query_frag" => Some(env_flag_default("GLRMASK_URI_QUERY_FRAG_TERMINAL", false)),
        "uri_query" => Some(env_flag_default("GLRMASK_URI_QUERY_TERMINAL", false)),
        "uri_fragment" => Some(env_flag_default("GLRMASK_URI_FRAGMENT_TERMINAL", false)),
        "uri_ipv6_address" => Some(env_flag_default("GLRMASK_URI_IPV6_ADDRESS_TERMINAL", false)),
        "uri_pct_encoded" => Some(env_flag_default("GLRMASK_URI_PCT_ENCODED_TERMINAL", false)),
        "uri_h16_colon" => Some(env_flag_default("GLRMASK_URI_H16_COLON_TERMINAL", false)),
        "uri_colon_h16" => Some(env_flag_default("GLRMASK_URI_COLON_H16_TERMINAL", false)),
        _ => None,
    }
}

fn uri_ipv6_alt_nonterminals() -> bool {
    env_flag("GLRMASK_URI_IPV6_ALT_NONTERMINALS")
}

fn uri_run_chunk_max() -> usize {
    std::env::var("GLRMASK_URI_RUN_CHUNK_MAX")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(24)
}

fn uri_charclass_run_regex(def: &str, max_run: usize) -> String {
    format!(r#"(?:[{def}]{{1,{max_run}}})"#)
}

fn json_wrapped_string_length_regex(min_len: usize, max_len: usize) -> String {
    let inner = if min_len == max_len {
        format!(r#"(?:{}){{{}}}"# , JSON_STRING_CHAR_PATTERN, min_len)
    } else {
        format!(
            r#"(?:{}){{{},{}}}"#,
            JSON_STRING_CHAR_PATTERN,
            min_len,
            max_len,
        )
    };
    string_value_body_regex(&inner)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SupportedMultipleOf {
    Integer(u64),
    ReciprocalPowerOfTen(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum IntegerMultipleState {
    Start,
    Sign,
    Zero,
    Integer(u64),
    Dot(u64),
    FractionZero(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum ReciprocalPow10State {
    Start,
    Sign,
    Zero,
    Integer,
    Dot,
    FractionWithin(usize),
    FractionBeyond,
}

fn supported_multiple_of(value: f64) -> Option<SupportedMultipleOf> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }

    let rounded = value.round();
    if (value - rounded).abs() <= 1e-9 && rounded >= 1.0 && rounded <= u64::MAX as f64 {
        return Some(SupportedMultipleOf::Integer(rounded as u64));
    }

    power_of_ten_multiple_scale(value).map(SupportedMultipleOf::ReciprocalPowerOfTen)
}

fn power_of_ten_multiple_scale(value: f64) -> Option<usize> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }

    let mut scaled = value;
    for scale in 0..=12 {
        if (scaled - 1.0).abs() <= 1e-9 {
            return Some(scale);
        }
        scaled *= 10.0;
    }
    None
}

fn integer_multiple_expr(multiple: u64, allow_fractional_zero: bool) -> LexerExpr {
    assert!(multiple > 0);

    SchemaCtx::build_state_machine_expr(
        IntegerMultipleState::Start,
        |state| match state {
            IntegerMultipleState::Zero => true,
            IntegerMultipleState::Integer(remainder) => remainder == 0,
            IntegerMultipleState::FractionZero(remainder) => remainder == 0,
            _ => false,
        },
        |state| {
            let mut transitions = Vec::new();
            match state {
                IntegerMultipleState::Start => {
                    transitions.push((b'-', IntegerMultipleState::Sign));
                    transitions.push((b'0', IntegerMultipleState::Zero));
                    for digit in b'1'..=b'9' {
                        let remainder = (digit - b'0') as u64 % multiple;
                        transitions.push((digit, IntegerMultipleState::Integer(remainder)));
                    }
                }
                IntegerMultipleState::Sign => {
                    transitions.push((b'0', IntegerMultipleState::Zero));
                    for digit in b'1'..=b'9' {
                        let remainder = (digit - b'0') as u64 % multiple;
                        transitions.push((digit, IntegerMultipleState::Integer(remainder)));
                    }
                }
                IntegerMultipleState::Zero => {
                    if allow_fractional_zero {
                        transitions.push((b'.', IntegerMultipleState::Dot(0)));
                    }
                }
                IntegerMultipleState::Integer(remainder) => {
                    if allow_fractional_zero {
                        transitions.push((b'.', IntegerMultipleState::Dot(remainder)));
                    }
                    for digit in b'0'..=b'9' {
                        let next = (remainder * 10 + (digit - b'0') as u64) % multiple;
                        transitions.push((digit, IntegerMultipleState::Integer(next)));
                    }
                }
                IntegerMultipleState::Dot(remainder) => {
                    transitions.push((b'0', IntegerMultipleState::FractionZero(remainder)));
                }
                IntegerMultipleState::FractionZero(remainder) => {
                    transitions.push((b'0', IntegerMultipleState::FractionZero(remainder)));
                }
            }
            transitions
        },
    )
}

fn reciprocal_power_of_ten_expr(scale: usize) -> LexerExpr {
    assert!(scale > 0);

    SchemaCtx::build_state_machine_expr(
        ReciprocalPow10State::Start,
        |state| match state {
            ReciprocalPow10State::Zero
            | ReciprocalPow10State::Integer
            | ReciprocalPow10State::FractionWithin(_)
            | ReciprocalPow10State::FractionBeyond => true,
            _ => false,
        },
        |state| {
            let mut transitions = Vec::new();
            match state {
                ReciprocalPow10State::Start => {
                    transitions.push((b'-', ReciprocalPow10State::Sign));
                    transitions.push((b'0', ReciprocalPow10State::Zero));
                    for digit in b'1'..=b'9' {
                        transitions.push((digit, ReciprocalPow10State::Integer));
                    }
                }
                ReciprocalPow10State::Sign => {
                    transitions.push((b'0', ReciprocalPow10State::Zero));
                    for digit in b'1'..=b'9' {
                        transitions.push((digit, ReciprocalPow10State::Integer));
                    }
                }
                ReciprocalPow10State::Zero | ReciprocalPow10State::Integer => {
                    transitions.push((b'.', ReciprocalPow10State::Dot));
                    if state == ReciprocalPow10State::Integer {
                        for digit in b'0'..=b'9' {
                            transitions.push((digit, ReciprocalPow10State::Integer));
                        }
                    }
                }
                ReciprocalPow10State::Dot => {
                    for digit in b'0'..=b'9' {
                        transitions.push((digit, ReciprocalPow10State::FractionWithin(1)));
                    }
                }
                ReciprocalPow10State::FractionWithin(consumed) => {
                    if consumed < scale {
                        for digit in b'0'..=b'9' {
                            transitions.push((digit, ReciprocalPow10State::FractionWithin(consumed + 1)));
                        }
                    } else {
                        transitions.push((b'0', ReciprocalPow10State::FractionBeyond));
                    }
                }
                ReciprocalPow10State::FractionBeyond => {
                    transitions.push((b'0', ReciprocalPow10State::FractionBeyond));
                }
            }
            transitions
        },
    )
}

fn compile_regex_union_expr(regexes: &[String]) -> LexerExpr {
    let exprs = regexes
        .iter()
        .map(|regex| parse_regex(regex, true))
        .collect::<Vec<_>>();
    if exprs.len() == 1 {
        exprs.into_iter().next().unwrap()
    } else {
        LexerExpr::Choice(exprs)
    }
}

/// Build a GrammarExpr body for a string value, fusing non-split quotes
/// into the body.  Returns `(terminal_body, wrapper)`.
///
///  - `terminal_body`: the expression to pass to `extract_terminal_rule`.
///    Non-split quotes are fused into it.
///  - `wrapper`: a closure that wraps the named terminal ref with the
///    split-off quote literals.
///
/// This is the GrammarExpr analogue of `string_value_body_regex` +
/// `wrap_string_value_terminal` for cases where the body is an
/// expression tree rather than a regex string.
fn wrap_string_value_expr_parts(body: GrammarExpr) -> (GrammarExpr, Box<dyn FnOnce(GrammarExpr) -> GrammarExpr>) {
    let open = split_open_quote();
    let close = split_close_quote();
    // Fuse non-split quotes into the body terminal
    let terminal_body = {
        let mut inner_parts = Vec::new();
        if !open { inner_parts.push(literal_expr(b"\"")); }
        inner_parts.push(body);
        if !close { inner_parts.push(literal_expr(b"\"")); }
        sequence_or_single(inner_parts)
    };
    // Wrap the terminal ref with split-off quotes
    let wrap = move |term: GrammarExpr| -> GrammarExpr {
        wrap_string_value_terminal(term)
    };
    (terminal_body, Box::new(wrap))
}

fn wrap_key_colon_expr_parts(body: GrammarExpr) -> (GrammarExpr, Box<dyn FnOnce(GrammarExpr) -> GrammarExpr>) {
    let open = split_open_quote();
    let close = split_close_quote();
    let colon = split_colon_space();
    let terminal_body = {
        let mut inner_parts = Vec::new();
        if !open {
            inner_parts.push(literal_expr(b"\""));
        }
        inner_parts.push(body);
        if !close {
            inner_parts.push(literal_expr(b"\""));
        }
        if !colon {
            inner_parts.push(key_colon_suffix_expr());
        }
        sequence_or_single(inner_parts)
    };
    let wrap = move |term: GrammarExpr| -> GrammarExpr { wrap_key_colon_terminal(term) };
    (terminal_body, Box::new(wrap))
}

/// Wrap an arbitrary GrammarExpr body as a quoted string value.
///
/// Equivalent to composing `wrap_string_value_expr_parts` but for cases
/// where the body does not need to be named as a separate terminal rule.
fn quoted_expr(inner: GrammarExpr) -> GrammarExpr {
    let open = split_open_quote();
    let close = split_close_quote();
    // Fuse non-split quotes into body, then wrap with split quotes
    let mut body_parts = Vec::new();
    if !open { body_parts.push(literal_expr(b"\"")); }
    body_parts.push(inner);
    if !close { body_parts.push(literal_expr(b"\"")); }
    let body = sequence_or_single(body_parts);
    wrap_string_value_terminal(body)
}

fn json_date_body_expr() -> GrammarExpr {
    let year = parsed_regex_expr(r#"[0-9]{4}"#, true);
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
    let day_31 = parsed_regex_expr(r#"(?:0[1-9]|[12][0-9]|3[01])"#, true);
    let day_30 = parsed_regex_expr(r#"(?:0[1-9]|[12][0-9]|30)"#, true);
    let day_29 = parsed_regex_expr(r#"(?:0[1-9]|1[0-9]|2[0-9])"#, true);

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
            day_29,
        ]),
    ])
}

fn json_time_body_expr() -> GrammarExpr {
    let hour = parsed_regex_expr(r#"(?:[01][0-9]|2[0-3])"#, true);
    let minute = parsed_regex_expr(r#"[0-5][0-9]"#, true);
    let second = parsed_regex_expr(r#"(?:[0-5][0-9]|60)"#, true);
    let fraction = GrammarExpr::Optional(Box::new(sequence_or_single(vec![
        literal_expr(b"."),
        parsed_regex_expr(r#"[0-9]+"#, true),
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

fn json_hostname_label_pattern() -> &'static str {
    r#"[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"#
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
    let bytes = json_value_literal_bytes(value);
    if value.is_string() && bytes.len() >= 2 && bytes[0] == b'"' {
        // String literal: split into body + quote wrapping via hierarchy.
        let has_close = bytes.last() == Some(&b'"');
        let body_end = if has_close { bytes.len() - 1 } else { bytes.len() };
        let body_only = &bytes[1..body_end];

        // Build body with fused non-split quotes
        let open = split_open_quote();
        let close = split_close_quote();
        let mut body_bytes = Vec::new();
        if !open { body_bytes.push(b'"'); }
        body_bytes.extend_from_slice(body_only);
        if !close && has_close { body_bytes.push(b'"'); }
        let body = literal_expr(&body_bytes);

        // Wrap with split-off quotes
        wrap_string_value_terminal(body)
    } else {
        literal_expr(&bytes)
    }
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
        "required",
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

    // ── enum ──────────────────────────────────────────────────────────────────
    let enum1 = s1.get("enum").and_then(Value::as_array);
    let enum2 = s2.get("enum").and_then(Value::as_array);
    match (enum1, enum2) {
        (Some(left), Some(right)) => {
            let inter: Vec<Value> = left.iter().filter(|v| right.contains(v)).cloned().collect();
            merged.insert("enum".into(), Value::Array(inter));
        }
        (Some(values), None) => {
            merged.insert("enum".into(), Value::Array(values.clone()));
        }
        (None, Some(values)) => {
            merged.insert("enum".into(), Value::Array(values.clone()));
        }
        (None, None) => {}
    }

    let handled: BTreeSet<&'static str> = [
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
        "enum",
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

pub fn json_schema_to_grammar(schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
        .unwrap_or(false);

    let t0 = std::time::Instant::now();
    let schema: Value = serde_json::from_str(schema_json)
        .map_err(|err| GlrMaskError::GrammarParse(err.to_string()))?;
    let parse_json_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let mut named = schema_to_named_grammar(&schema)?;
    let schema_to_named_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let promote_enabled = std::env::var("GLRMASK_PROMOTE_LARGE_LITERAL_ALTS")
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off"))
        .unwrap_or(true);

    let t2 = std::time::Instant::now();
    if promote_enabled {
        promote_large_literal_alts(&mut named, 10);
    }
    let promote_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = std::time::Instant::now();
    let result = lower(&named)?;
    let lower_ms = t3.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][import] parse_json_ms={:.3} schema_to_named_ms={:.3} promote_ms={:.3} lower_ms={:.3} rules={} total_ms={:.3}",
            parse_json_ms, schema_to_named_ms, promote_ms, lower_ms, result.rules.len(),
            t0.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(result)
}

pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {
    let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
        .unwrap_or(false);

    let t0 = std::time::Instant::now();
    let mut ctx = SchemaCtx::new(schema);
    let new_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    ctx.register_root_definitions();
    let register_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = std::time::Instant::now();
    ctx.materialize_registered_refs()?;
    let materialize_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = std::time::Instant::now();
    let start_expr = match ctx.convert_schema(schema) {
        Ok(expr) => expr,
        Err(err) if is_unsat_schema_error(&err) => never_expr(),
        Err(err) => return Err(err),
    };
    let convert_ms = t3.elapsed().as_secs_f64() * 1000.0;

    ctx.insert_rule("start", start_expr);
    ctx.hoist_pattern_terminals_in_nonterminals();
    let terminal_names: BTreeSet<String> = ctx
        .rules
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| rule_name_is_terminal(name))
        .map(|s| s.to_string())
        .collect();
    let rules: Vec<NamedRule> = ctx.rules.into_iter().map(|(name, expr)| {
        let is_terminal = terminal_names.contains(&name);
        NamedRule { name, expr, is_terminal, is_internal: false }
    }).collect();

    // Mark terminal rules as internal-only when they are never referenced
    // from any nonterminal rule body.  Internal terminals exist solely as
    // sub-expressions of other terminal rules (resolved via Expr::Shared).
    let grammar_visible = collect_grammar_visible_refs(&rules, &terminal_names);
    let rules = rules.into_iter().map(|mut rule| {
        if rule.is_terminal
            && !grammar_visible.contains(&rule.name)
            && !rule_name_force_visible_terminal(&rule.name)
        {
            rule.is_internal = true;
        }
        rule
    }).collect();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][schema_to_named] new_ms={:.3} register_ms={:.3} materialize_ms={:.3} convert_ms={:.3} total_ms={:.3}",
            new_ms, register_ms, materialize_ms, convert_ms, t0.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(NamedGrammar {
        rules,
        start: "start".into(),
        ignore: None,
    })
}

struct SchemaCtx<'a> {
    root_schema: &'a Value,
    coerce_one_of: bool,
    lenient_json_schema: bool,
    rules: Vec<(String, GrammarExpr)>,
    rule_indices: BTreeMap<String, usize>,
    used_rule_names: BTreeSet<String>,
    ref_rule_names: BTreeMap<String, String>,
    ref_compile_stack: BTreeSet<String>,
    generated_object_rule_counter: usize,
    generated_rule_counter: usize,
    expr_dedup_cache: BTreeMap<String, String>,
    json_string_exact_cache: BTreeMap<usize, String>,
    json_string_upto_cache: BTreeMap<usize, String>,
    shared_ap_literal_keys: BTreeSet<String>,
    shared_ap_key_body_expr: Option<GrammarExpr>,
    shared_ap_key_body_rule_cache: BTreeMap<Vec<String>, String>,
    shared_ap_key_colon_expr: Option<GrammarExpr>,
    shared_ap_key_rule_cache: BTreeMap<Vec<String>, String>,
    draft_stack: Vec<JsonSchemaDraft>,
    convert_depth: usize,
    /// Cache for `convert_schema`: identical sub-schemas produce the same
    /// grammar expression, avoiding duplicate rule generation.
    schema_convert_cache: BTreeMap<String, GrammarExpr>,
}

fn rule_name_is_terminal(name: &str) -> bool {
    uri_rule_should_be_terminal(name).unwrap_or(!name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit()))
}

fn rule_name_force_visible_terminal(name: &str) -> bool {
    name.starts_with("URI_")
        || uri_rule_should_be_terminal(name) == Some(true)
}

impl<'a> SchemaCtx<'a> {
    fn new(root: &'a Value) -> Self {
        let (coerce_one_of, lenient_json_schema) = root
            .as_object()
            .and_then(|object| object.get("x-guidance"))
            .and_then(Value::as_object)
            .map(|guidance| {
                (
                    guidance
                        .get("coerce_one_of")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    guidance
                        .get("lenient")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                )
            })
            .unwrap_or((false, false));

        let mut ctx = Self {
            root_schema: root,
            coerce_one_of,
            lenient_json_schema,
            rules: Vec::new(),
            rule_indices: BTreeMap::new(),
            used_rule_names: BTreeSet::new(),
            ref_rule_names: BTreeMap::new(),
            ref_compile_stack: BTreeSet::new(),
            generated_object_rule_counter: 0,
            generated_rule_counter: 0,
            expr_dedup_cache: BTreeMap::new(),
            json_string_exact_cache: BTreeMap::new(),
            json_string_upto_cache: BTreeMap::new(),
            shared_ap_literal_keys: collect_shared_ap_literal_keys(root),
            shared_ap_key_body_expr: None,
            shared_ap_key_body_rule_cache: BTreeMap::new(),
            shared_ap_key_colon_expr: None,
            shared_ap_key_rule_cache: BTreeMap::new(),
            draft_stack: vec![DEFAULT_JSON_SCHEMA_DRAFT],
            convert_depth: 0,
            schema_convert_cache: BTreeMap::new(),
        };
        ctx.ensure_base_rules();
        ctx
    }

    fn try_append_suffix_to_trailing_literal_expr(
        &self,
        expr: GrammarExpr,
        suffix: &[u8],
    ) -> Option<GrammarExpr> {
        match expr {
            GrammarExpr::Ref(rule_name) => {
                let rule_expr = self
                    .rule_indices
                    .get(&rule_name)
                    .and_then(|&index| self.rules.get(index))?
                    .1
                    .clone();
                self.try_append_suffix_to_trailing_literal_expr(rule_expr, suffix)
            }
            other => try_append_suffix_to_trailing_literal(other, suffix),
        }
    }

    fn try_take_leading_container_delim_expr(
        &self,
        expr: GrammarExpr,
    ) -> Option<(u8, GrammarExpr)> {
        match expr {
            GrammarExpr::Ref(rule_name) => {
                let rule_expr = self
                    .rule_indices
                    .get(&rule_name)
                    .and_then(|&index| self.rules.get(index))?
                    .1
                    .clone();
                self.try_take_leading_container_delim_expr(rule_expr)
            }
            other => try_take_leading_container_delim(other),
        }
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

    fn one_of_is_explicitly_coerced(&self) -> bool {
        self.coerce_one_of || self.lenient_json_schema
    }

    fn schema_type_names<'b>(&self, schema: &'b Value) -> Option<BTreeSet<&'b str>> {
        let object = schema.as_object()?;
        match object.get("type") {
            Some(Value::String(type_name)) => Some(BTreeSet::from([type_name.as_str()])),
            Some(Value::Array(type_names)) => {
                let names = type_names
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<BTreeSet<_>>();
                if names.is_empty() {
                    None
                } else {
                    Some(names)
                }
            }
            _ => None,
        }
    }

    fn schema_object_property<'b>(&self, schema: &'b Value, key: &str) -> Option<&'b Value> {
        schema
            .as_object()?
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get(key))
    }

    fn schema_enum_values<'b>(&self, schema: &'b Value) -> Option<&'b [Value]> {
        schema
            .as_object()?
            .get("enum")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
    }

    fn schema_const_value<'b>(&self, schema: &'b Value) -> Option<&'b Value> {
        schema.as_object()?.get("const")
    }

    fn unescape_json_schema_pattern_literal(pattern: &str) -> Option<String> {
        let mut out = String::with_capacity(pattern.len());
        let mut chars = pattern.chars();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                let escaped = chars.next()?;
                match escaped {
                    '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^'
                    | '$' | '|' | '-' => out.push(escaped),
                    _ => return None,
                }
            } else if matches!(ch, '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|')
            {
                return None;
            } else {
                out.push(ch);
            }
        }
        Some(out)
    }

    fn schema_exact_string_literal<'b>(&self, schema: &'b Value) -> Option<String> {
        let object = schema.as_object()?;

        if let Some(Value::String(value)) = object.get("const") {
            return Some(value.clone());
        }

        let types = self.schema_type_names(schema);
        if let Some(types) = &types {
            if !types.contains("string") {
                return None;
            }
        }

        let pattern = object.get("pattern")?.as_str()?;
        let anchored = pattern.strip_prefix('^')?.strip_suffix('$')?;
        Self::unescape_json_schema_pattern_literal(anchored)
    }

    fn resolved_structural_options(
        &self,
        schema: &Map<String, Value>,
        keyword: &str,
    ) -> Option<Vec<Value>> {
        let options = schema.get(keyword)?.as_array()?;
        let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
        Some(
            options
                .iter()
                .map(|option| {
                    if base.is_empty() {
                        Value::Object(self.schema_for_intersection(option))
                    } else {
                        Value::Object(
                            self.merge_resolved_subschemas(&base, std::slice::from_ref(option)),
                        )
                    }
                })
                .collect(),
        )
    }

    fn normalized_allof_schema(&self, schema: &Map<String, Value>) -> Option<Value> {
        let all_of = schema.get("allOf")?.as_array()?;
        let base = Self::schema_without_keys(schema, &["allOf"]);
        Some(Value::Object(self.merge_resolved_subschemas(&base, all_of)))
    }

    fn schemas_are_verifiably_disjoint(&self, left: &Value, right: &Value) -> bool {
        if left == &Value::Bool(false) || right == &Value::Bool(false) {
            return true;
        }
        if left == &Value::Bool(true) || right == &Value::Bool(true) {
            return false;
        }

        let Some(left_object) = left.as_object() else {
            return false;
        };
        let Some(right_object) = right.as_object() else {
            return false;
        };

        if let Some(reference) = left_object.get("$ref").and_then(Value::as_str) {
            if let Ok(target) = self.resolve_local_ref(reference) {
                return self.schemas_are_verifiably_disjoint(target, right);
            }
            return false;
        }
        if let Some(reference) = right_object.get("$ref").and_then(Value::as_str) {
            if let Ok(target) = self.resolve_local_ref(reference) {
                return self.schemas_are_verifiably_disjoint(left, target);
            }
            return false;
        }

        if let Some(normalized) = self.normalized_allof_schema(left_object) {
            return self.schemas_are_verifiably_disjoint(&normalized, right);
        }
        if let Some(normalized) = self.normalized_allof_schema(right_object) {
            return self.schemas_are_verifiably_disjoint(left, &normalized);
        }

        if let (Some(left_const), Some(right_const)) = (
            self.schema_const_value(left),
            self.schema_const_value(right),
        ) {
            return left_const != right_const;
        }

        if let (Some(left_const), Some(right_enum)) = (
            self.schema_const_value(left),
            self.schema_enum_values(right),
        ) {
            return !right_enum.iter().any(|value| value == left_const);
        }

        if let (Some(left_enum), Some(right_const)) = (
            self.schema_enum_values(left),
            self.schema_const_value(right),
        ) {
            return !left_enum.iter().any(|value| value == right_const);
        }

        if let (Some(left_enum), Some(right_enum)) = (
            self.schema_enum_values(left),
            self.schema_enum_values(right),
        ) {
            return left_enum
                .iter()
                .all(|left_value| !right_enum.iter().any(|right_value| right_value == left_value));
        }

        if let (Some(left_literal), Some(right_literal)) = (
            self.schema_exact_string_literal(left),
            self.schema_exact_string_literal(right),
        ) {
            return left_literal != right_literal;
        }

        if let Some(any_of) = self.resolved_structural_options(left_object, "anyOf") {
            return any_of
                .iter()
                .all(|option| self.schemas_are_verifiably_disjoint(option, right));
        }
        if let Some(one_of) = self.resolved_structural_options(left_object, "oneOf") {
            return one_of
                .iter()
                .all(|option| self.schemas_are_verifiably_disjoint(option, right));
        }
        if let Some(any_of) = self.resolved_structural_options(right_object, "anyOf") {
            return any_of
                .iter()
                .all(|option| self.schemas_are_verifiably_disjoint(left, option));
        }
        if let Some(one_of) = self.resolved_structural_options(right_object, "oneOf") {
            return one_of
                .iter()
                .all(|option| self.schemas_are_verifiably_disjoint(left, option));
        }

        let left_types = self.schema_type_names(left);
        let right_types = self.schema_type_names(right);
        if let (Some(left_types), Some(right_types)) = (&left_types, &right_types) {
            let overlaps = left_types.iter().any(|left_type| {
                right_types.iter().any(|right_type| {
                    left_type == right_type
                        || (*left_type == "integer" && *right_type == "number")
                        || (*left_type == "number" && *right_type == "integer")
                })
            });
            if !overlaps {
                return true;
            }
        }

        let left_is_object = left_types
            .as_ref()
            .map(|types| types.contains("object"))
            .unwrap_or_else(|| {
                left_object.contains_key("properties") || left_object.contains_key("required")
            });
        let right_is_object = right_types
            .as_ref()
            .map(|types| types.contains("object"))
            .unwrap_or_else(|| {
                right_object.contains_key("properties") || right_object.contains_key("required")
            });
        if left_is_object && right_is_object {
            let mut keys = BTreeSet::new();
            if let Some(required) = left_object.get("required").and_then(Value::as_array) {
                keys.extend(required.iter().filter_map(Value::as_str).map(String::from));
            }
            if let Some(required) = right_object.get("required").and_then(Value::as_array) {
                keys.extend(required.iter().filter_map(Value::as_str).map(String::from));
            }

            return keys.into_iter().any(|key| {
                let left_prop = self
                    .schema_object_property(left, &key)
                    .unwrap_or(&Value::Bool(true));
                let right_prop = self
                    .schema_object_property(right, &key)
                    .unwrap_or(&Value::Bool(true));
                self.schemas_are_verifiably_disjoint(left_prop, right_prop)
            });
        }

        false
    }

    fn one_of_options_are_safe_to_coerce(
        &self,
        schema: &Map<String, Value>,
        options: &[Value],
    ) -> bool {
        let resolved: Vec<Value> = options
            .iter()
            .map(|option| {
                if has_structural_keywords(schema) {
                    let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
                    Value::Object(self.merge_resolved_subschemas(&base, std::slice::from_ref(option)))
                } else {
                    Value::Object(self.schema_for_intersection(option))
                }
            })
            .collect();

        resolved.iter().enumerate().all(|(index, left)| {
            resolved
                .iter()
                .skip(index + 1)
                .all(|right| self.schemas_are_verifiably_disjoint(left, right))
        })
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

    fn insert_shared_rule(&mut self, name: impl Into<String>, expr: GrammarExpr) -> String {
        let key = expr_key(&expr);
        if let Some(rule_name) = self.expr_dedup_cache.get(&key) {
            return rule_name.clone();
        }

        let name = self.insert_rule(name, expr);
        self.expr_dedup_cache.insert(key, name.clone());
        name
    }

    fn insert_named_terminal_rule(&mut self, name: impl Into<String>, expr: GrammarExpr) -> GrammarExpr {
        let name = name.into();
        self.insert_rule(name.clone(), expr);
        GrammarExpr::Ref(name)
    }

    fn insert_uri_rule(&mut self, name: &str, expr: GrammarExpr) -> GrammarExpr {
        if uri_rule_should_be_terminal(name).unwrap_or(false) {
            self.insert_named_terminal_rule(name.to_string(), expr)
        } else {
            self.insert_rule(name.to_string(), expr);
            GrammarExpr::Ref(name.into())
        }
    }

    fn extract_pattern_terminal_rule(&mut self, expr: GrammarExpr, prefix: &str) -> GrammarExpr {
        let key = expr_key(&expr);
        if let Some(rule_name) = self.expr_dedup_cache.get(&key) {
            return GrammarExpr::Ref(rule_name.clone());
        }

        let rule_name = self.fresh_rule_name(prefix);
        self.insert_rule(rule_name.clone(), expr);
        self.expr_dedup_cache.insert(key, rule_name.clone());
        GrammarExpr::Ref(rule_name)
    }

    fn hoist_patterns_in_expr(&mut self, expr: GrammarExpr, prefix: &str) -> GrammarExpr {
        match expr {
            GrammarExpr::Ref(_) | GrammarExpr::Literal(_) | GrammarExpr::Epsilon => expr,
            GrammarExpr::CharClass { .. } | GrammarExpr::RawRegex(_) | GrammarExpr::AnyByte => {
                self.extract_pattern_terminal_rule(expr, prefix)
            }
            GrammarExpr::Sequence(parts) => sequence_or_single(
                parts
                    .into_iter()
                    .map(|part| self.hoist_patterns_in_expr(part, prefix))
                    .collect(),
            ),
            GrammarExpr::Choice(parts) => choice_or_single(
                parts
                    .into_iter()
                    .map(|part| self.hoist_patterns_in_expr(part, prefix))
                    .collect(),
            ),
            GrammarExpr::Optional(inner) => {
                GrammarExpr::Optional(Box::new(self.hoist_patterns_in_expr(*inner, prefix)))
            }
            GrammarExpr::Repeat(inner) => {
                GrammarExpr::Repeat(Box::new(self.hoist_patterns_in_expr(*inner, prefix)))
            }
            GrammarExpr::RepeatOne(inner) => {
                GrammarExpr::RepeatOne(Box::new(self.hoist_patterns_in_expr(*inner, prefix)))
            }
            GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
                expr: Box::new(self.hoist_patterns_in_expr(*expr, prefix)),
                min,
                max,
            },
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.hoist_patterns_in_expr(*expr, prefix)),
                exclude: Box::new(self.hoist_patterns_in_expr(*exclude, prefix)),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.hoist_patterns_in_expr(*expr, prefix)),
                intersect: Box::new(self.hoist_patterns_in_expr(*intersect, prefix)),
            },
            GrammarExpr::SeparatedSequence { items, separator, allow_empty } => GrammarExpr::SeparatedSequence {
                items: items
                    .into_iter()
                    .map(|(item_expr, required)| {
                        (self.hoist_patterns_in_expr(item_expr, prefix), required)
                    })
                    .collect(),
                separator: Box::new(self.hoist_patterns_in_expr(*separator, prefix)),
                allow_empty,
            },
        }
    }

    fn hoist_pattern_terminals_in_nonterminals(&mut self) {
        let rule_names: Vec<String> = self.rules.iter().map(|(name, _)| name.clone()).collect();
        for rule_name in rule_names {
            if rule_name_is_terminal(&rule_name) {
                continue;
            }
            let Some(&index) = self.rule_indices.get(&rule_name) else {
                continue;
            };
            let expr = self.rules[index].1.clone();
            self.rules[index].1 = self.hoist_patterns_in_expr(expr, "INLINE_PATTERN");
        }
    }

    fn fresh_rule_name(&mut self, prefix: &str) -> String {
        let prefix = sanitize_rule_name(prefix);
        loop {
            let name = format!("{}_{}", prefix, self.generated_rule_counter);
            self.generated_rule_counter += 1;
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

                let expr = GrammarExpr::RepeatRange {
                    expr: Box::new(self.json_string_char_ref()),
                    min: count,
                    max: count,
                };

                let rule = self.extract_terminal_rule(expr, &format!("JSON_STRING_CHAR_EXACT_{count}"));
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

                let expr = GrammarExpr::RepeatRange {
                    expr: Box::new(self.json_string_char_ref()),
                    min: 0,
                    max,
                };

                let rule = self.extract_terminal_rule(expr, &format!("JSON_STRING_CHAR_UPTO_{max}"));
                if let GrammarExpr::Ref(rule_name) = &rule {
                    self.json_string_upto_cache.insert(max, rule_name.clone());
                }
                rule
            }
        }
    }

    fn should_split_bounded_string(&self, min_len: usize, max_len: Option<usize>) -> bool {
        let chunk = json_string_repeat_chunk();
        min_len > chunk
            || max_len
                .map(|value| value > chunk)
                .unwrap_or(false)
    }

    fn build_split_json_string_exact_expr(&mut self, count: usize) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if count == 0 {
            return empty_expr();
        }
        if count <= chunk {
            return self.json_string_char_exact_ref(count);
        }

        let full_chunks = count / chunk;
        let remainder = count % chunk;
        let mut parts = Vec::new();
        if full_chunks == 1 {
            parts.push(self.json_string_char_exact_ref(chunk));
        } else if full_chunks > 1 {
            parts.push(repeat_expr(
                self.json_string_char_exact_ref(chunk),
                full_chunks,
                Some(full_chunks),
            ));
        }
        if remainder > 0 {
            parts.push(self.json_string_char_exact_ref(remainder));
        }
        sequence_or_single(parts)
    }

    fn build_split_json_string_upto_expr(&mut self, max: usize) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if max == 0 {
            return empty_expr();
        }
        if max <= chunk {
            return self.json_string_char_upto_ref(max);
        }

        let full_chunks = max / chunk;
        let remainder = max % chunk;
        let mut options = Vec::new();

        if full_chunks == 1 {
            options.push(self.json_string_char_upto_ref(chunk));
        } else {
            options.push(sequence_or_single(vec![
                repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    0,
                    Some(full_chunks - 1),
                ),
                self.json_string_char_upto_ref(chunk),
            ]));
        }

        if remainder > 0 {
            options.push(sequence_or_single(vec![
                self.build_split_json_string_exact_expr(full_chunks * chunk),
                self.json_string_char_upto_ref(remainder),
            ]));
        }

        choice_or_single(options)
    }

    fn build_split_json_string_body(
        &mut self,
        min_len: usize,
        max_len: Option<usize>,
    ) -> GrammarExpr {
        let mut parts = Vec::new();
        if min_len > 0 {
            parts.push(self.build_split_json_string_exact_expr(min_len));
        }
        match max_len {
            Some(max_len) => {
                if max_len > min_len {
                    parts.push(self.build_split_json_string_upto_expr(max_len - min_len));
                }
            }
            None => parts.push(repeat_expr(self.json_string_char_ref(), 0, None)),
        }
        sequence_or_single(parts)
    }

    /// Like `build_split_json_string_body`, but fuses a suffix (e.g. closing `"`)
    /// into the last terminal chunk so it doesn't become a standalone terminal.
    fn build_split_json_string_body_with_suffix(
        &mut self,
        min_len: usize,
        max_len: Option<usize>,
        suffix: GrammarExpr,
    ) -> GrammarExpr {
        match max_len {
            Some(max_len) if max_len == min_len => {
                if min_len == 0 {
                    return suffix;
                }
                self.build_split_json_string_exact_expr_with_suffix(min_len, suffix)
            }
            Some(max_len) => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(self.build_split_json_string_exact_expr(min_len));
                }
                if max_len > min_len {
                    parts.push(self.build_split_json_string_upto_expr_with_suffix(
                        max_len - min_len,
                        suffix,
                    ));
                }
                sequence_or_single(parts)
            }
            None => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(self.build_split_json_string_exact_expr(min_len));
                }
                let repeat_close = self.extract_terminal_rule(
                    sequence_or_single(vec![
                        GrammarExpr::Repeat(Box::new(self.json_string_char_ref())),
                        suffix,
                    ]),
                    "JSON_STRING_REPEAT_CLOSE",
                );
                parts.push(repeat_close);
                sequence_or_single(parts)
            }
        }
    }

    /// Like `build_split_json_string_upto_expr`, but the last leaf terminal
    /// in each choice alternative includes a fused suffix.
    fn build_split_json_string_upto_expr_with_suffix(
        &mut self,
        max: usize,
        suffix: GrammarExpr,
    ) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if max == 0 {
            return suffix;
        }
        if max <= chunk {
            let upto = self.json_string_char_upto_ref(max);
            return self.extract_terminal_rule(
                sequence_or_single(vec![upto, suffix]),
                "JSON_STRING_CHAR_UPTO_CLOSE",
            );
        }

        let full_chunks = max / chunk;
        let remainder = max % chunk;
        let mut options = Vec::new();

        let upto_chunk_close = {
            let upto = self.json_string_char_upto_ref(chunk);
            self.extract_terminal_rule(
                sequence_or_single(vec![upto, suffix.clone()]),
                "JSON_STRING_CHAR_UPTO_CLOSE",
            )
        };

        if full_chunks == 1 {
            options.push(upto_chunk_close);
        } else {
            options.push(sequence_or_single(vec![
                repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    0,
                    Some(full_chunks - 1),
                ),
                upto_chunk_close,
            ]));
        }

        if remainder > 0 {
            let upto_rem_close = {
                let upto = self.json_string_char_upto_ref(remainder);
                self.extract_terminal_rule(
                    sequence_or_single(vec![upto, suffix]),
                    "JSON_STRING_CHAR_UPTO_CLOSE",
                )
            };
            options.push(sequence_or_single(vec![
                self.build_split_json_string_exact_expr(full_chunks * chunk),
                upto_rem_close,
            ]));
        }

        choice_or_single(options)
    }

    /// Like `build_split_json_string_exact_expr`, but the last terminal chunk
    /// includes a fused suffix.
    fn build_split_json_string_exact_expr_with_suffix(
        &mut self,
        count: usize,
        suffix: GrammarExpr,
    ) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if count == 0 {
            return suffix;
        }
        if count <= chunk {
            let exact = self.json_string_char_exact_ref(count);
            return self.extract_terminal_rule(
                sequence_or_single(vec![exact, suffix]),
                "JSON_STRING_CHAR_EXACT_CLOSE",
            );
        }

        let full_chunks = count / chunk;
        let remainder = count % chunk;
        let mut parts = Vec::new();

        if remainder > 0 {
            if full_chunks == 1 {
                parts.push(self.json_string_char_exact_ref(chunk));
            } else {
                parts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    full_chunks,
                    Some(full_chunks),
                ));
            }
            let exact_rem = self.json_string_char_exact_ref(remainder);
            parts.push(self.extract_terminal_rule(
                sequence_or_single(vec![exact_rem, suffix]),
                "JSON_STRING_CHAR_EXACT_CLOSE",
            ));
        } else {
            if full_chunks == 1 {
                let exact = self.json_string_char_exact_ref(chunk);
                return self.extract_terminal_rule(
                    sequence_or_single(vec![exact, suffix]),
                    "JSON_STRING_CHAR_EXACT_CLOSE",
                );
            }
            parts.push(repeat_expr(
                self.json_string_char_exact_ref(chunk),
                full_chunks - 1,
                Some(full_chunks - 1),
            ));
            let exact_last = self.json_string_char_exact_ref(chunk);
            parts.push(self.extract_terminal_rule(
                sequence_or_single(vec![exact_last, suffix]),
                "JSON_STRING_CHAR_EXACT_CLOSE",
            ));
        }

        sequence_or_single(parts)
    }

    /// Like `build_split_json_string_exact_expr`, but fuses a prefix with the
    /// first terminal-sized chunk (≤ 1024 chars).
    fn build_split_json_string_exact_expr_with_prefix(
        &mut self,
        count: usize,
        prefix: GrammarExpr,
    ) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if count == 0 {
            return prefix;
        }
        if count <= chunk {
            let exact = self.json_string_char_exact_ref(count);
            return self.extract_terminal_rule(
                sequence_or_single(vec![prefix, exact]),
                "JSON_STRING_CHAR_EXACT_OPEN",
            );
        }

        let full_chunks = count / chunk;
        let remainder = count % chunk;

        let first_exact = self.json_string_char_exact_ref(chunk);
        let first_open = self.extract_terminal_rule(
            sequence_or_single(vec![prefix, first_exact]),
            "JSON_STRING_CHAR_EXACT_OPEN",
        );

        let mut parts = vec![first_open];
        if full_chunks > 1 {
            parts.push(repeat_expr(
                self.json_string_char_exact_ref(chunk),
                full_chunks - 1,
                Some(full_chunks - 1),
            ));
        }
        if remainder > 0 {
            parts.push(self.json_string_char_exact_ref(remainder));
        }

        sequence_or_single(parts)
    }

    /// Like `build_split_json_string_upto_expr`, but fuses a prefix with the
    /// first terminal-sized chunk in each choice alternative.
    fn build_split_json_string_upto_expr_with_prefix(
        &mut self,
        max: usize,
        prefix: GrammarExpr,
    ) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if max == 0 {
            return prefix;
        }
        if max <= chunk {
            let upto = self.json_string_char_upto_ref(max);
            return self.extract_terminal_rule(
                sequence_or_single(vec![prefix, upto]),
                "JSON_STRING_CHAR_UPTO_OPEN",
            );
        }

        let full_chunks = max / chunk;
        let remainder = max % chunk;
        let mut options = Vec::new();

        // Subcase: 0 exact reps → just upto_1024 with prefix
        let upto_chunk = self.json_string_char_upto_ref(chunk);
        let upto_chunk_open = self.extract_terminal_rule(
            sequence_or_single(vec![prefix.clone(), upto_chunk]),
            "JSON_STRING_CHAR_UPTO_OPEN",
        );

        if full_chunks == 1 {
            options.push(upto_chunk_open);
        } else {
            // 0 reps → prefix + upto_1024
            options.push(upto_chunk_open);
            // 1+ reps → exact_1024_open + Repeat(exact_1024, 0, N-2) + upto_1024
            let exact_chunk = self.json_string_char_exact_ref(chunk);
            let exact_open = self.extract_terminal_rule(
                sequence_or_single(vec![prefix.clone(), exact_chunk]),
                "JSON_STRING_CHAR_EXACT_OPEN",
            );
            let mut subparts = vec![exact_open];
            if full_chunks > 2 {
                subparts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    0,
                    Some(full_chunks - 2),
                ));
            }
            subparts.push(self.json_string_char_upto_ref(chunk));
            options.push(sequence_or_single(subparts));
        }

        if remainder > 0 {
            let exact_chunk = self.json_string_char_exact_ref(chunk);
            let exact_open = self.extract_terminal_rule(
                sequence_or_single(vec![prefix.clone(), exact_chunk]),
                "JSON_STRING_CHAR_EXACT_OPEN",
            );
            let mut subparts = vec![exact_open];
            if full_chunks > 1 {
                subparts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    full_chunks - 1,
                    Some(full_chunks - 1),
                ));
            }
            subparts.push(self.json_string_char_upto_ref(remainder));
            options.push(sequence_or_single(subparts));
        }

        choice_or_single(options)
    }

    /// Like `build_split_json_string_upto_expr`, but fuses both a prefix with
    /// the first chunk and a suffix with the last chunk in each alternative.
    fn build_split_json_string_upto_expr_with_prefix_and_suffix(
        &mut self,
        max: usize,
        prefix: GrammarExpr,
        suffix: GrammarExpr,
    ) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if max == 0 {
            return sequence_or_single(vec![prefix, suffix]);
        }
        if max <= chunk {
            let upto = self.json_string_char_upto_ref(max);
            return self.extract_terminal_rule(
                sequence_or_single(vec![prefix, upto, suffix]),
                "JSON_STRING_CHAR_UPTO_WRAPPED",
            );
        }

        let full_chunks = max / chunk;
        let remainder = max % chunk;
        let mut options = Vec::new();

        // Subcase: 0 exact reps → prefix + upto_1024 + suffix
        let upto_chunk = self.json_string_char_upto_ref(chunk);
        let upto_wrapped = self.extract_terminal_rule(
            sequence_or_single(vec![prefix.clone(), upto_chunk, suffix.clone()]),
            "JSON_STRING_CHAR_UPTO_WRAPPED",
        );

        if full_chunks == 1 {
            options.push(upto_wrapped);
        } else {
            options.push(upto_wrapped);

            // 1+ reps → exact_open + middle + upto_close
            let exact_chunk = self.json_string_char_exact_ref(chunk);
            let exact_open = self.extract_terminal_rule(
                sequence_or_single(vec![prefix.clone(), exact_chunk]),
                "JSON_STRING_CHAR_EXACT_OPEN",
            );
            let upto_chunk2 = self.json_string_char_upto_ref(chunk);
            let upto_close = self.extract_terminal_rule(
                sequence_or_single(vec![upto_chunk2, suffix.clone()]),
                "JSON_STRING_CHAR_UPTO_CLOSE",
            );
            let mut subparts = vec![exact_open];
            if full_chunks > 2 {
                subparts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    0,
                    Some(full_chunks - 2),
                ));
            }
            subparts.push(upto_close);
            options.push(sequence_or_single(subparts));
        }

        if remainder > 0 {
            let exact_chunk = self.json_string_char_exact_ref(chunk);
            let exact_open = self.extract_terminal_rule(
                sequence_or_single(vec![prefix.clone(), exact_chunk]),
                "JSON_STRING_CHAR_EXACT_OPEN",
            );
            let upto_rem = self.json_string_char_upto_ref(remainder);
            let upto_rem_close = self.extract_terminal_rule(
                sequence_or_single(vec![upto_rem, suffix.clone()]),
                "JSON_STRING_CHAR_UPTO_CLOSE",
            );
            let mut subparts = vec![exact_open];
            if full_chunks > 1 {
                subparts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    full_chunks - 1,
                    Some(full_chunks - 1),
                ));
            }
            subparts.push(upto_rem_close);
            options.push(sequence_or_single(subparts));
        }

        choice_or_single(options)
    }

    /// General split body builder with optional prefix/suffix fusion.
    fn build_split_json_string_body_wrapped(
        &mut self,
        min_len: usize,
        max_len: Option<usize>,
        prefix: Option<GrammarExpr>,
        suffix: Option<GrammarExpr>,
    ) -> GrammarExpr {
        let has_exact = min_len > 0;
        let has_tail = match max_len {
            Some(ml) if ml > min_len => true,
            None => true,
            _ => false,
        };

        match (has_exact, has_tail) {
            (true, true) => {
                // Two parts: exact + tail. Prefix on exact, suffix on tail.
                let exact = if let Some(p) = prefix {
                    self.build_split_json_string_exact_expr_with_prefix(min_len, p)
                } else {
                    self.build_split_json_string_exact_expr(min_len)
                };

                let tail = match max_len {
                    Some(ml) => {
                        if let Some(s) = suffix {
                            self.build_split_json_string_upto_expr_with_suffix(ml - min_len, s)
                        } else {
                            self.build_split_json_string_upto_expr(ml - min_len)
                        }
                    }
                    None => {
                        if let Some(s) = suffix {
                            self.extract_terminal_rule(
                                sequence_or_single(vec![
                                    GrammarExpr::Repeat(Box::new(self.json_string_char_ref())),
                                    s,
                                ]),
                                "JSON_STRING_REPEAT_CLOSE",
                            )
                        } else {
                            repeat_expr(self.json_string_char_ref(), 0, None)
                        }
                    }
                };
                sequence_or_single(vec![exact, tail])
            }
            (true, false) => {
                // Exact only (min == max). Both prefix and suffix go here.
                match (prefix, suffix) {
                    (Some(p), Some(s)) => {
                        self.build_split_json_string_exact_expr_with_prefix_and_suffix(min_len, p, s)
                    }
                    (Some(p), None) => {
                        self.build_split_json_string_exact_expr_with_prefix(min_len, p)
                    }
                    (None, Some(s)) => {
                        self.build_split_json_string_exact_expr_with_suffix(min_len, s)
                    }
                    (None, None) => {
                        self.build_split_json_string_exact_expr(min_len)
                    }
                }
            }
            (false, true) => {
                // Tail only (min == 0). Both prefix and suffix go on tail.
                match max_len {
                    Some(ml) => match (prefix, suffix) {
                        (Some(p), Some(s)) => {
                            self.build_split_json_string_upto_expr_with_prefix_and_suffix(ml, p, s)
                        }
                        (Some(p), None) => {
                            self.build_split_json_string_upto_expr_with_prefix(ml, p)
                        }
                        (None, Some(s)) => {
                            self.build_split_json_string_upto_expr_with_suffix(ml, s)
                        }
                        (None, None) => {
                            self.build_split_json_string_upto_expr(ml)
                        }
                    },
                    None => {
                        let mut inner = Vec::new();
                        if let Some(p) = prefix { inner.push(p); }
                        inner.push(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
                        if let Some(s) = suffix { inner.push(s); }
                        self.extract_terminal_rule(
                            sequence_or_single(inner),
                            "JSON_STRING_REPEAT_WRAPPED",
                        )
                    }
                }
            }
            (false, false) => {
                // min=0, max=0: body is empty
                let mut inner = Vec::new();
                if let Some(p) = prefix { inner.push(p); }
                if let Some(s) = suffix { inner.push(s); }
                if inner.is_empty() { empty_expr() } else { sequence_or_single(inner) }
            }
        }
    }

    /// Like `build_split_json_string_exact_expr`, but fuses both a prefix with
    /// the first chunk and a suffix with the last chunk.
    fn build_split_json_string_exact_expr_with_prefix_and_suffix(
        &mut self,
        count: usize,
        prefix: GrammarExpr,
        suffix: GrammarExpr,
    ) -> GrammarExpr {
        let chunk = json_string_repeat_chunk();
        if count == 0 {
            return sequence_or_single(vec![prefix, suffix]);
        }
        if count <= chunk {
            let exact = self.json_string_char_exact_ref(count);
            return self.extract_terminal_rule(
                sequence_or_single(vec![prefix, exact, suffix]),
                "JSON_STRING_CHAR_EXACT_WRAPPED",
            );
        }

        let full_chunks = count / chunk;
        let remainder = count % chunk;

        let first_exact = self.json_string_char_exact_ref(chunk);
        let first_open = self.extract_terminal_rule(
            sequence_or_single(vec![prefix, first_exact]),
            "JSON_STRING_CHAR_EXACT_OPEN",
        );

        let mut parts = vec![first_open];

        if remainder > 0 {
            if full_chunks > 1 {
                parts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    full_chunks - 1,
                    Some(full_chunks - 1),
                ));
            }
            let last_exact = self.json_string_char_exact_ref(remainder);
            parts.push(self.extract_terminal_rule(
                sequence_or_single(vec![last_exact, suffix]),
                "JSON_STRING_CHAR_EXACT_CLOSE",
            ));
        } else {
            if full_chunks > 2 {
                parts.push(repeat_expr(
                    self.json_string_char_exact_ref(chunk),
                    full_chunks - 2,
                    Some(full_chunks - 2),
                ));
            }
            let last_exact = self.json_string_char_exact_ref(chunk);
            parts.push(self.extract_terminal_rule(
                sequence_or_single(vec![last_exact, suffix]),
                "JSON_STRING_CHAR_EXACT_CLOSE",
            ));
        }

        sequence_or_single(parts)
    }

    fn json_key_colon_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_KEY_COLON_RULE.into())
    }

    /// Full key-colon expression using the hierarchy, for use in
    /// DFA-level operations (Exclude, DFA building) where a terminal-compilable
    /// expression is needed.
    fn json_key_colon_full_expr() -> GrammarExpr {
        regex_expr(key_colon_body_regex(JSON_STRING_BODY_ONLY_REGEX))
    }

    fn json_integer_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_INTEGER_RULE.into())
    }

    fn json_number_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_NUMBER_RULE.into())
    }

    fn json_number_type_expr(&self) -> GrammarExpr {
        choice_or_single(vec![self.json_integer_ref(), self.json_number_ref()])
    }

    fn json_bool_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_BOOL_RULE.into())
    }

    fn json_null_ref(&self) -> GrammarExpr {
        GrammarExpr::Ref(JSON_NULL_RULE.into())
    }

    fn ensure_base_rules(&mut self) {
        self.insert_rule(JSON_STRING_CHAR_RULE, regex_expr(JSON_STRING_CHAR_PATTERN));

        // JSON_STRING_BODY_RULE: split-aware body terminal used by json_string
        // so the nonterminal does not embed the character body directly.
        let (body_expr, _) = wrap_string_value_expr_parts(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
        self.insert_rule(JSON_STRING_BODY_RULE, body_expr);

        // JSON_STRING_MIDDLE_RULE: reusable middle fragment used to build
        // readable pattern-property key terminals.
        self.insert_rule(JSON_STRING_MIDDLE_RULE, GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
        self.insert_rule(
            JSON_STRING_MIDDLE_END_RULE,
            sequence_or_single(vec![
                GrammarExpr::Ref(JSON_STRING_MIDDLE_RULE.into()),
                literal_expr(b"\""),
            ]),
        );

        // JSON_STRING_RULE: always assembled from literals + named terminals.
        let json_string_expr = wrap_string_value_terminal(GrammarExpr::Ref(JSON_STRING_BODY_RULE.into()));
        self.insert_rule(JSON_STRING_RULE, json_string_expr);
        self.insert_rule(JSON_INTEGER_RULE, regex_expr(r#"-?(0|[1-9][0-9]*)"#));
        self.insert_rule(JSON_NUMBER_RULE, regex_expr(JSON_NUMBER_NONINTEGER_REGEX));
        self.insert_rule(JSON_NONNEG_INTEGER_RULE, regex_expr(r#"(0|[1-9][0-9]*)"#));
        self.insert_rule(JSON_NONNEG_NUMBER_RULE, regex_expr(JSON_NONNEG_NUMBER_NONINTEGER_REGEX));
        self.insert_rule(
            JSON_BOOL_RULE,
            choice_or_single(vec![literal_expr(b"true"), literal_expr(b"false")]),
        );
        self.insert_rule(JSON_NULL_RULE, literal_expr(b"null"));

        // JSON_KEY_COLON_BODY_RULE: split-aware key-colon body terminal.
        let (kc_body_expr, _) = wrap_key_colon_expr_parts(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
        self.insert_rule(JSON_KEY_COLON_BODY_RULE, kc_body_expr);

        // JSON_KEY_COLON_RULE: always assembled from literals + named terminals.
        let json_key_colon_expr = wrap_key_colon_terminal(GrammarExpr::Ref(JSON_KEY_COLON_BODY_RULE.into()));
        self.insert_rule(JSON_KEY_COLON_RULE, json_key_colon_expr);
        self.insert_rule(
            JSON_KV_RULE,
            sequence_or_single(vec![self.json_key_colon_ref(), self.json_value_ref()]),
        );
        self.insert_rule(
            JSON_OBJECT_RULE,
            sequence_or_single(vec![
                literal_expr(b"{"),
                GrammarExpr::SeparatedSequence {
                    items: vec![(
                        GrammarExpr::Repeat(Box::new(GrammarExpr::Ref(JSON_KV_RULE.into()))),
                        true,
                    )],
                    separator: Box::new(self.json_item_separator_expr()),
                    allow_empty: true,
                },
                literal_expr(b"}"),
            ]),
        );
        self.insert_rule(
            JSON_ARRAY_RULE,
            sequence_or_single(vec![
                literal_expr(b"["),
                GrammarExpr::SeparatedSequence {
                    items: vec![(GrammarExpr::Repeat(Box::new(self.json_value_ref())), true)],
                    separator: Box::new(self.json_item_separator_expr()),
                    allow_empty: true,
                },
                literal_expr(b"]"),
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

    fn resolve_local_ref(&self, ref_value: &str) -> Result<&Value, GlrMaskError> {
        if !ref_value.starts_with('#') {
            return Err(GlrMaskError::GrammarParse(format!(
                "unsupported $ref '{ref_value}'"
            )));
        }

        if ref_value == "#" {
            return Ok(self.root_schema);
        }

        if !ref_value.starts_with("#/") {
            return find_local_ref_target(self.root_schema, ref_value)
                .ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!("unknown $ref target '{ref_value}'"))
                });
        }

        let mut current = self.root_schema;
        for token in ref_value[2..].split('/') {
            let key = decode_local_ref_token(token);
            current = current.get(&key).ok_or_else(|| {
                GlrMaskError::GrammarParse(format!("unknown $ref target '{ref_value}'"))
            })?;
        }
        Ok(current)
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
        let siblings = Self::schema_without_keys(object, &["$ref"]);
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

    fn schema_without_keys(
        schema: &Map<String, Value>,
        excluded: &[&str],
    ) -> Map<String, Value> {
        schema
            .iter()
            .filter(|(key, _)| !excluded.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn convert_schema_options(
        &mut self,
        options: &[Value],
    ) -> Result<Vec<GrammarExpr>, GlrMaskError> {
        options
            .iter()
            .map(|option| self.convert_schema(option))
            .collect()
    }

    fn exact_closed_object_disabled() -> bool {
        std::env::var("GLRMASK_DISABLE_EXACT_CLOSED_OBJECT_UNION")
            .map(|v| {
                let n = v.trim().to_ascii_lowercase();
                !matches!(n.as_str(), "" | "0" | "false" | "no" | "off")
            })
            .unwrap_or(false)
    }

    /// Returns true when the separator-merged left-recursive grammar for
    /// closed objects with optional keys is enabled.  Off by default; opt-in
    /// via `GLRMASK_ENABLE_FACTORED_CLOSED_OBJECT=1`.
    fn factored_closed_object_enabled() -> bool {
        env_flag("GLRMASK_ENABLE_FACTORED_CLOSED_OBJECT")
    }

    fn collect_ordered_closed_object_schema_variant(
        &mut self,
        variant: &Map<String, Value>,
        allow_implicit_object: bool,
    ) -> Result<Option<OrderedClosedObjectSchemaVariant>, GlrMaskError> {
        // By default require explicit `type: object`. Without it, object keywords
        // only constrain object instances and non-objects remain valid.
        // For oneOf, a union of multiple such closed-object branches still rejects
        // non-objects because all branches match them, so the exact closed-object
        // union path may safely intercept that specific implicit-object pattern.
        match variant.get("type").and_then(Value::as_str) {
            Some("object") => {}
            None if allow_implicit_object
                && variant.contains_key("properties")
                && !variant.contains_key("patternProperties")
                && !variant.contains_key("propertyNames") => {}
            _ => return Ok(None),
        }
        match variant.get("additionalProperties") {
            Some(Value::Bool(false)) => {}
            _ => return Ok(None),
        }
        if variant.contains_key("patternProperties")
            || variant.contains_key("propertyNames")
            || variant.contains_key("minProperties")
            || variant.contains_key("maxProperties")
        {
            return Ok(None);
        }

        let properties = variant
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let required: BTreeSet<String> = variant
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).map(String::from).collect())
            .unwrap_or_default();

        if required.iter().any(|key| !properties.contains_key(key)) {
            return Ok(None);
        }

        let items = properties
            .into_iter()
            .map(|(key, value_schema)| OrderedClosedObjectSchemaItem {
                required: required.contains(&key),
                key,
                value_schema,
            })
            .collect();
        Ok(Some(OrderedClosedObjectSchemaVariant { items }))
    }

    fn ordered_subset_accept_count(
        variants: &[OrderedClosedObjectVariant],
        state: &[OrderedSubsetCursor],
    ) -> usize {
        state
            .iter()
            .filter(|entry| {
                variants[entry.variant_idx as usize].close_allowed(entry.cursor as usize)
            })
            .count()
    }

    fn ordered_subset_close_allowed(
        mode: StructuralBranchMode,
        variants: &[OrderedClosedObjectVariant],
        state: &[OrderedSubsetCursor],
    ) -> bool {
        let accepting = Self::ordered_subset_accept_count(variants, state);
        match mode {
            StructuralBranchMode::AnyOf => accepting > 0,
            StructuralBranchMode::OneOf => accepting == 1,
        }
    }

    fn ordered_subset_legal_next_keys(
        variants: &[OrderedClosedObjectVariant],
        state: &[OrderedSubsetCursor],
    ) -> Vec<String> {
        let mut keys = BTreeSet::new();
        for entry in state {
            for key in variants[entry.variant_idx as usize]
                .legal_next_keys(entry.cursor as usize)
            {
                keys.insert(key.to_string());
            }
        }
        keys.into_iter().collect()
    }

    fn ordered_subset_transition(
        variants: &[OrderedClosedObjectVariant],
        state: &[OrderedSubsetCursor],
        key: &str,
    ) -> Vec<OrderedSubsetCursor> {
        let mut next = Vec::new();
        for entry in state {
            if let Some(next_cursor) = variants[entry.variant_idx as usize]
                .advance_cursor(entry.cursor as usize, key)
            {
                next.push(OrderedSubsetCursor {
                    variant_idx: entry.variant_idx,
                    cursor: next_cursor as u16,
                });
            }
        }
        next.sort_unstable();
        next.dedup();
        next
    }

    fn build_exact_ordered_closed_object_variants(
        &mut self,
        variants: Vec<OrderedClosedObjectVariant>,
        mode: StructuralBranchMode,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if variants.is_empty() || variants.len() > exact_closed_object_union_max_variants() {
            return Ok(None);
        }
        let key_count = variants
            .iter()
            .flat_map(|variant| variant.items.iter().map(|item| item.key.as_str()))
            .collect::<BTreeSet<_>>()
            .len();
        if key_count > exact_closed_object_union_max_keys() {
            return Ok(None);
        }

        let start_state: Vec<OrderedSubsetCursor> = (0..variants.len())
            .map(|variant_idx| OrderedSubsetCursor {
                variant_idx: variant_idx as u16,
                cursor: 0,
            })
            .collect();
        let mut states = vec![start_state.clone()];
        let mut transitions: Vec<Vec<(String, GrammarExpr, usize)>> = vec![Vec::new()];
        let mut state_to_idx = BTreeMap::new();
        state_to_idx.insert(start_state, 0usize);
        let mut queue = VecDeque::from([0usize]);

        while let Some(state_idx) = queue.pop_front() {
            let state = states[state_idx].clone();
            let mut edges = Vec::new();
            for key in Self::ordered_subset_legal_next_keys(&variants, &state) {
                let mut grouped_next_states: Vec<(GrammarExpr, Vec<OrderedSubsetCursor>)> = Vec::new();
                for entry in &state {
                    let variant = &variants[entry.variant_idx as usize];
                    let Some(next_cursor) = variant.advance_cursor(entry.cursor as usize, &key) else {
                        continue;
                    };
                    let Some(value_expr) = variant.value_expr_for_key(&key) else {
                        continue;
                    };
                    if let Some((_, next_state)) = grouped_next_states
                        .iter_mut()
                        .find(|(existing_expr, _)| *existing_expr == value_expr)
                    {
                        next_state.push(OrderedSubsetCursor {
                            variant_idx: entry.variant_idx,
                            cursor: next_cursor as u16,
                        });
                    } else {
                        grouped_next_states.push((
                            value_expr,
                            vec![OrderedSubsetCursor {
                                variant_idx: entry.variant_idx,
                                cursor: next_cursor as u16,
                            }],
                        ));
                    }
                }

                for (value_expr, mut next_state) in grouped_next_states {
                    next_state.sort_unstable();
                    next_state.dedup();
                    if next_state.is_empty() {
                        continue;
                    }
                    let next_idx = if let Some(&idx) = state_to_idx.get(&next_state) {
                        idx
                    } else {
                        let idx = states.len();
                        if idx >= exact_closed_object_union_max_states() {
                            return Ok(None);
                        }
                        states.push(next_state.clone());
                        transitions.push(Vec::new());
                        state_to_idx.insert(next_state, idx);
                        queue.push_back(idx);
                        idx
                    };
                    edges.push((key.clone(), value_expr, next_idx));
                }
            }
            transitions[state_idx] = edges;
        }

        let mut reverse_edges: Vec<Vec<usize>> = vec![Vec::new(); states.len()];
        for (state_idx, edges) in transitions.iter().enumerate() {
            for (_, _, next_idx) in edges {
                reverse_edges[*next_idx].push(state_idx);
            }
        }

        let mut viable = vec![false; states.len()];
        let mut viable_queue = VecDeque::new();
        for (state_idx, state) in states.iter().enumerate() {
            if Self::ordered_subset_close_allowed(mode, &variants, state) {
                viable[state_idx] = true;
                viable_queue.push_back(state_idx);
            }
        }

        while let Some(state_idx) = viable_queue.pop_front() {
            for &prev_idx in &reverse_edges[state_idx] {
                if viable[prev_idx] {
                    continue;
                }
                viable[prev_idx] = true;
                viable_queue.push_back(prev_idx);
            }
        }

        if !viable[0] {
            return Ok(None);
        }

        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_ord_q_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_p1")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;
        let state_rule_names: Vec<String> = (0..states.len())
            .map(|idx| format!("{base_name}_p{idx}"))
            .collect();

        // Emit the exact-union DFA as forward continuation states.
        // This keeps each consumed prefix in a single grammar state, rather than
        // encoding the same suffix transition through multiple historical prefixes.
        for state_idx in 0..states.len() {
            if !viable[state_idx] {
                continue;
            }

            let mut alts = Vec::new();
            if Self::ordered_subset_close_allowed(mode, &variants, &states[state_idx]) {
                alts.push(empty_expr());
            }

            for (key, value_expr, next_idx) in &transitions[state_idx] {
                if !viable[*next_idx] {
                    continue;
                }

                let sep: &[u8] = if state_idx == 0 { b"" } else { b", " };
                alts.push(sequence_or_single(vec![
                    self.build_fused_merged_literal_key_value_expr(
                        sep,
                        key,
                        value_expr.clone(),
                    ),
                    GrammarExpr::Ref(state_rule_names[*next_idx].clone()),
                ]));
            }

            if alts.is_empty() {
                return Ok(None);
            }
            self.insert_rule(state_rule_names[state_idx].clone(), choice_or_single(alts));
        }

        Ok(Some(sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(state_rule_names[0].clone()),
            literal_expr(b"}"),
        ])))
    }

    fn try_build_exact_closed_object(
        &mut self,
        ordered: &[(String, GrammarExpr, bool)],
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if ordered.is_empty() || ordered.len() > exact_closed_object_single_max_keys() {
            return Ok(None);
        }

        let variant = OrderedClosedObjectVariant {
            items: ordered
                .iter()
                .map(|(key, value_expr, required)| OrderedClosedObjectItem {
                    key: key.clone(),
                    value_expr: value_expr.clone(),
                    required: *required,
                })
                .collect(),
        };

        self.build_exact_ordered_closed_object_variants(
            vec![variant],
            StructuralBranchMode::AnyOf,
        )
    }

    /// Build a factored DFA for an open object with ordered known keys and
    /// an additional-properties continuation.  The DFA eliminates the O(log N)
    /// path duplication of the binary tree approach, reducing max_paths from
    /// ~log2(N)+1 to ~2 (known-key vs AP during key parsing only).
    ///
    /// Grammar structure:
    ///   Object → "{" body[0] "}" | "{" free_nc "}"   (when all-optional)
    ///   body[j] → KV(k_i) after[i+1]  for each legal key k_i from state j
    ///   after[j] → ", " dispatch[j] | ε   (when close allowed, factored sep)
    ///   after[j] → ", " body[j]           (when close not allowed)
    ///   dispatch[j] → body[j]             (ordered — key disambiguates)
    ///               | PP_pair free_c      (pattern property — key disambig.)
    ///               | AP_pair additional_c (additional property — key disambig.)
    /// Build a left-recursive grammar for ordered objects.
    ///
    /// The key insight: merge the separator `", "` into the key literal for all
    /// non-first keys. This makes each key transition look like:
    ///   prefix[k] → prefix[j] SEP_KV_k
    /// where SEP_KV_k = `", "key_k": ` is a SINGLE merged literal terminal.
    ///
    /// Because each SEP_KV_k token is unique (different key string), the parser
    /// is 100% deterministic after any prefix[j]: it simply matches the next token.
    ///
    /// Grammar:
    ///   First-key rules (no separator in literal):
    ///     prefix[k] → KV_k          if k is reachable from state 0 (gap 0..k all optional)
    ///   Continuation rules (separator merged into literal):
    ///     prefix[k] → prefix[j] SEP_KV_k   for each j where gap j..k all optional, j > 0
    ///
    ///   body → prefix[j]              for each closeable j
    ///         | prefix[j] free_expr   for closeable j with free properties
    ///         | free_expr              if can close at 0 (all optional + has free)
    ///         | ε                      if can close at 0
    ///
    ///   Object → "{" body "}"
    ///
    /// Stack depth: O(1) at all times — prefix reduces eagerly after each key.
    /// Determinism: after prefix[j], the next token uniquely identifies the next rule.
    ///   - If next token is SEP_KV_k: shift and reduce prefix[k] → prefix[j] SEP_KV_k
    ///   - If next token is `}`: reduce body → prefix[j]
    ///   - If next token is `, ` (not merged): this is the free-property separator
    ///     In this case there IS a small fork at `, `: prefix[j] (close or continue).
    ///     But with the merged literal, no key starts with `, ` as a bare separator — 
    ///     the separator IS the key literal. So `, "k1": ` is ONE token.
    ///     The only ambiguity remains at the BODY level for free properties.
    fn try_build_factored_ordered_object(
        &mut self,
        ordered: &[(String, GrammarExpr, bool)],
        free_c: &str,
        free_nc: &str,
        free_pair_exprs: &[GrammarExpr],
        additional_pair_exprs: &[GrammarExpr],
        additional_c: &str,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if ordered.is_empty() || ordered.len() > factored_open_object_max_keys() {
            return Ok(None);
        }

        let n = ordered.len();
        let has_free = !free_pair_exprs.is_empty() || !additional_pair_exprs.is_empty();

        // Allocate base name.
        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_lrec_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_p1")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;

        let prefix_name = |j: usize| -> String { format!("{base_name}_p{j}") };

        // Helper: build free-property expression.
        let make_free_expr = |free_pair_exprs: &[GrammarExpr],
                               additional_pair_exprs: &[GrammarExpr],
                               free_c: &str,
                               additional_c: &str| -> GrammarExpr {
            let mut alts = Vec::new();
            for pp in free_pair_exprs {
                alts.push(sequence_or_single(vec![
                    pp.clone(),
                    GrammarExpr::Ref(free_c.to_string()),
                ]));
            }
            for ap in additional_pair_exprs {
                alts.push(sequence_or_single(vec![
                    ap.clone(),
                    GrammarExpr::Ref(additional_c.to_string()),
                ]));
            }
            choice_or_single(alts)
        };

        // State j can close iff all remaining keys ordered[j..n] are optional.
        let can_close_at = |j: usize| -> bool {
            ordered[j..].iter().all(|(_, _, req)| !req)
        };

        // Emit prefix rules: prefix[j] for j in 1..=n.
        // prefix[j] represents any valid ordered prefix ending at key k_{j-1}.
        // KEY TECHNIQUE: for continuation keys (i > 0), merge the separator into
        // the literal so that `, "key_j": ` is a SINGLE token. This makes the
        // grammar deterministic: after prefix[i], the parser sees a unique token
        // for each possible next key, eliminating nondeterminism at ", ".
        for j in 1..=n {
            let (key_j, value_j, _) = &ordered[j - 1];
            let mut alts = Vec::new();

            for i in 0..j {
                // gap i..(j-1) must all be optional.
                let gap_ok = ordered[i..j - 1].iter().all(|(_, _, req)| !req);
                if !gap_ok { continue; }

                if i == 0 {
                    // First key: no leading separator in the literal.
                    let kv = if has_free && ordered.len() >= 16 {
                        self.build_fused_merged_literal_key_value_expr(
                            b"",
                            key_j,
                            value_j.clone(),
                        )
                    } else {
                        self.build_merged_literal_key_value_expr(
                            b"",
                            key_j,
                            value_j.clone(),
                        )
                    };
                    alts.push(kv);
                } else {
                    // Continuation key: MERGE the separator `", "` into the key literal.
                    // This turns `, "key_j": ` into a single token, making the
                    // grammar deterministic after prefix[i].
                    let sep_kv = self.build_fused_merged_literal_key_value_expr(
                        b", ",
                        key_j,
                        value_j.clone(),
                    );
                    alts.push(sequence_or_single(vec![
                        GrammarExpr::Ref(prefix_name(i)),
                        sep_kv,
                    ]));
                }
            }

            if alts.is_empty() { return Ok(None); }
            self.insert_rule(prefix_name(j), choice_or_single(alts));
        }

        // Build body alternatives.
        // Each closeable state j contributes:
        //   - prefix[j]                     (close after known keys)
        //   - prefix[j] ", " free_expr       (if has_free: continue with free properties)
        // State 0 (no known keys yet) contributes:
        //   - free_expr                      (if has_free)
        //   - ε                              (if can close at 0)
        let mut body_alts = Vec::new();

        for j in 1..=n {
            if !can_close_at(j) { continue; }
            body_alts.push(GrammarExpr::Ref(prefix_name(j)));
            if has_free {
                let free_expr = make_free_expr(
                    free_pair_exprs, additional_pair_exprs, free_c, additional_c
                );
                body_alts.push(sequence_or_single(vec![
                    GrammarExpr::Ref(prefix_name(j)),
                    self.json_item_separator_expr(),
                    free_expr,
                ]));
            }
        }

        if can_close_at(0) {
            if has_free {
                let free_expr = make_free_expr(
                    free_pair_exprs, additional_pair_exprs, free_c, additional_c
                );
                body_alts.push(free_expr);
            }
            body_alts.push(empty_expr());
        }

        if body_alts.is_empty() { return Ok(None); }

        let all_optional = ordered.iter().all(|(_, _, req)| !req);
        let inner = sequence_or_single(vec![
            literal_expr(b"{"),
            choice_or_single(body_alts),
            literal_expr(b"}"),
        ]);

        if all_optional && !free_nc.is_empty() {
            Ok(Some(choice_or_single(vec![
                inner,
                sequence_or_single(vec![
                    literal_expr(b"{"),
                    GrammarExpr::Ref(free_nc.to_string()),
                    literal_expr(b"}"),
                ]),
            ])))
        } else {
            Ok(Some(inner))
        }
    }

    fn try_build_exact_closed_object_union(
        &mut self,
        schema: &Map<String, Value>,
        options: &[Value],
        mode: StructuralBranchMode,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if options.len() < 2 || options.len() > exact_closed_object_union_max_variants() {
            return Ok(None);
        }

        let resolved: Vec<Map<String, Value>> = options
            .iter()
            .map(|opt| {
                if has_structural_keywords(schema) {
                    let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
                    self.merge_resolved_subschemas(&base, std::slice::from_ref(opt))
                } else {
                    self.schema_for_intersection(opt)
                }
            })
            .collect();

        let mut schema_variants = Vec::with_capacity(resolved.len());
        for variant in &resolved {
            let Some(ordered) = self.collect_ordered_closed_object_schema_variant(
                variant,
                mode == StructuralBranchMode::OneOf,
            )? else {
                return Ok(None);
            };
            schema_variants.push(ordered);
        }

        let variants: Vec<OrderedClosedObjectVariant> = schema_variants
            .into_iter()
            .map(|variant| {
                let mut items = Vec::new();
                for item in variant.items {
                    match self.convert_schema(&item.value_schema) {
                        Ok(value_expr) => items.push(OrderedClosedObjectItem {
                            value_expr,
                            key: item.key,
                            required: item.required,
                        }),
                        Err(err) if is_unsat_schema_error(&err) => {
                            if item.required {
                                return Err(unsat_schema_error());
                            }
                        }
                        Err(err) => return Err(err),
                    }
                }
                Ok(OrderedClosedObjectVariant { items })
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.build_exact_ordered_closed_object_variants(variants, mode)
    }

    /// Try to merge anyOf/oneOf variants that are all closed objects
    /// (additionalProperties: false) with shared + mutually-exclusive unique
    /// properties into a single ordering tree with a one-shot unique-kv
    /// continuation.  Returns `None` when the guard conditions are not met.
    fn try_merge_anyof_closed_objects(
        &mut self,
        schema: &Map<String, Value>,
        options: &[Value],
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if options.len() < 2 {
            return Ok(None);
        }

        // Resolve each variant to a concrete object schema.
        let resolved: Vec<Map<String, Value>> = options
            .iter()
            .map(|opt| {
                if has_structural_keywords(schema) {
                    let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
                    self.merge_resolved_subschemas(&base, std::slice::from_ref(opt))
                } else {
                    self.schema_for_intersection(opt)
                }
            })
            .collect();

        // Guard 1: every variant must be a plain object with explicit
        //          "type": "object", additionalProperties: false, no patternProperties.
        for v in &resolved {
            if v.get("type").and_then(Value::as_str) != Some("object") {
                return Ok(None);
            }
            match v.get("additionalProperties") {
                Some(Value::Bool(false)) => {}
                _ => return Ok(None),
            }
            if v.contains_key("patternProperties") {
                return Ok(None);
            }
            if v.contains_key("minProperties") || v.contains_key("maxProperties") {
                return Ok(None);
            }
            if v.contains_key("propertyNames") {
                return Ok(None);
            }
        }

        // Collect each variant's properties map and required set.
        let mut variant_props: Vec<&Map<String, Value>> = Vec::new();
        let mut variant_required: Vec<BTreeSet<String>> = Vec::new();
        for v in &resolved {
            let props = match v.get("properties").and_then(Value::as_object) {
                Some(p) => p,
                None => return Ok(None),
            };
            variant_props.push(props);
            let req: BTreeSet<String> = v
                .get("required")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().filter_map(Value::as_str).map(String::from).collect())
                .unwrap_or_default();
            variant_required.push(req);
        }

        // Classify every property as shared or unique.
        // shared: present in ALL variants with identical value schemas.
        // unique: present in exactly ONE variant.
        let all_keys: BTreeSet<String> = variant_props
            .iter()
            .flat_map(|p| p.keys().cloned())
            .collect();

        let mut shared_keys: Vec<String> = Vec::new();
        let mut unique_keys: Vec<(String, usize)> = Vec::new(); // (key, variant_index)

        for key in &all_keys {
            let present_in: Vec<usize> = variant_props
                .iter()
                .enumerate()
                .filter(|(_, p)| p.contains_key(key))
                .map(|(i, _)| i)
                .collect();

            if present_in.len() == variant_props.len() {
                // Guard: identical value schema across all variants.
                let first_schema = &variant_props[0][key];
                for &idx in &present_in[1..] {
                    if &variant_props[idx][key] != first_schema {
                        return Ok(None);
                    }
                }
                shared_keys.push(key.clone());
            } else if present_in.len() == 1 {
                unique_keys.push((key.clone(), present_in[0]));
            } else {
                // Present in some but not all → cannot merge.
                return Ok(None);
            }
        }

        // Guard: shared-property required status must be identical across all
        // variants.
        for key in &shared_keys {
            let first_req = variant_required[0].contains(key);
            for req_set in &variant_required[1..] {
                if req_set.contains(key) != first_req {
                    return Ok(None);
                }
            }
        }

        // Guard: unique properties must not be required in their variant.
        // (Other variants lack the property entirely, so the merged form
        //  must keep it optional.)
        for (key, vi) in &unique_keys {
            if variant_required[*vi].contains(key) {
                return Ok(None);
            }
        }

        // ---- All guards passed; build the merged expression. ----

        // 1. Shared ordered properties.
        let shared_required: BTreeSet<&String> = shared_keys
            .iter()
            .filter(|k| variant_required[0].contains(*k))
            .collect();
        let mut ordered: Vec<(String, GrammarExpr, bool)> = Vec::new();
        for key in &shared_keys {
            let value_schema = &variant_props[0][key];
            let value_expr = self.convert_schema(value_schema)?;
            ordered.push((key.clone(), value_expr, shared_required.contains(key)));
        }

        // 2. Unique kv-pair choice (one-shot optional).
        let unique_kv_exprs: Vec<GrammarExpr> = unique_keys
            .iter()
            .map(|(key, vi)| {
                let value_schema = &variant_props[*vi][key];
                let value_expr = self.convert_schema(value_schema)?;
                Ok(self.build_merged_literal_key_value_expr(b"", key, value_expr))
            })
            .collect::<Result<Vec<_>, GlrMaskError>>()?;

        // 3. Allocate an obj_ord base name.
        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_ord_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_t0")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;

        // 4. Build the shared ordering tree.
        if ordered.is_empty() && unique_kv_exprs.is_empty() {
            return Ok(Some(sequence_or_single(vec![
                literal_expr(b"{"),
                literal_expr(b"}"),
            ])));
        }

        let (tree_expr, tree_can_be_empty) =
            self.build_ordered_object_body_separated_sequence_expr(&ordered, true);

        // Helper: wrap an object-only expression with non-object alternatives
        // when the resolved variants don't restrict to type: "object".
        let all_typed_object = resolved.iter().all(|v| {
            v.get("type").and_then(Value::as_str) == Some("object")
        });
        let wrap_with_type_alternatives = |this: &mut Self, expr: GrammarExpr| -> GrammarExpr {
            if all_typed_object {
                expr
            } else {
                let mut alts = vec![expr];
                alts.push(this.json_array_ref());
                alts.push(this.json_string_ref());
                alts.push(this.json_number_type_expr());
                alts.push(this.json_bool_ref());
                alts.push(this.json_null_ref());
                choice_or_single(alts)
            }
        };

        // 5. Compose with unique continuation.
        if unique_kv_exprs.is_empty() {
            // No unique properties — just the shared tree.
            let body = sequence_or_single(vec![literal_expr(b"{"), tree_expr, literal_expr(b"}")]);
            if tree_can_be_empty {
                let obj_expr = choice_or_single(vec![
                    body,
                    sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]),
                ]);
                return Ok(Some(wrap_with_type_alternatives(self, obj_expr)));
            }
            return Ok(Some(wrap_with_type_alternatives(self, body)));
        }

        let unique_kv_choice = choice_or_single(unique_kv_exprs);

        // unique_c: ", " unique_kv | ε  (after tree content)
        let unique_c_name = format!("{base_name}_uc");
        self.insert_rule(
            unique_c_name.clone(),
            choice_or_single(vec![
                sequence_or_single(vec![
                    self.json_item_separator_expr(),
                    unique_kv_choice.clone(),
                ]),
                empty_expr(),
            ]),
        );

        // unique_nc: unique_kv | ε  (after '{', no preceding content)
        let unique_nc_name = format!("{base_name}_unc");
        self.insert_rule(
            unique_nc_name.clone(),
            choice_or_single(vec![unique_kv_choice, empty_expr()]),
        );

        let tree_prefix = sequence_or_single(vec![literal_expr(b"{"), tree_expr]);
        let with_tree = sequence_or_single(vec![
            tree_prefix,
            GrammarExpr::Ref(unique_c_name),
            literal_expr(b"}"),
        ]);

        let object_expr = if tree_can_be_empty {
            choice_or_single(vec![
                with_tree,
                sequence_or_single(vec![
                    literal_expr(b"{"),
                    GrammarExpr::Ref(unique_nc_name),
                    literal_expr(b"}"),
                ]),
            ])
        } else {
            with_tree
        };

        Ok(Some(wrap_with_type_alternatives(self, object_expr)))
    }

    /// For `anyOf` of open objects, detect when one variant's language dominates the union
    /// and compile just that variant to eliminate GLR ambiguity.
    ///
    /// For open objects (no `additionalProperties: false`), `language(A) ⊆ language(B)` iff:
    ///   1. `B.required ⊆ A.required`  (B requires fewer keys → accepts more objects)
    ///   2. `B.properties.keys() ⊆ A.properties.keys()`  (B has fewer typed constraints)
    ///   3. For every key `k` shared by both: `B.properties[k] == A.properties[k]`
    ///
    /// When these hold for all `i ≠ j`, `anyOf[A₀…Aₙ] = B` exactly, so compiling just `B`
    /// is language-preserving and eliminates the ambiguous GLR branching.
    fn try_reduce_anyof_open_objects(
        &mut self,
        schema: &Map<String, Value>,
        options: &[Value],
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if options.len() < 2 {
            return Ok(None);
        }

        // Resolve each option to a concrete schema map (owned).
        let resolved: Vec<Map<String, Value>> = options
            .iter()
            .map(|opt| {
                if has_structural_keywords(schema) {
                    let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
                    self.merge_resolved_subschemas(&base, std::slice::from_ref(opt))
                } else {
                    self.schema_for_intersection(opt)
                }
            })
            .collect();

        // Guard: every variant must be a plain open object with no complex sub-keywords.
        for v in &resolved {
            if v.get("type").and_then(Value::as_str) != Some("object") {
                return Ok(None);
            }
            if matches!(v.get("additionalProperties"), Some(Value::Bool(false))) {
                return Ok(None);
            }
            for key in &["patternProperties", "anyOf", "oneOf", "allOf", "$ref",
                         "minProperties", "maxProperties", "if", "then", "else"] {
                if v.contains_key(*key) {
                    return Ok(None);
                }
            }
        }

        // Extract (required_keys, property_keys) as owned Vecs for each variant.
        let info: Vec<(Vec<String>, Vec<String>)> = resolved
            .iter()
            .map(|v| {
                let required: Vec<String> = v
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).collect())
                    .unwrap_or_default();
                let prop_keys: Vec<String> = v
                    .get("properties")
                    .and_then(Value::as_object)
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                (required, prop_keys)
            })
            .collect();

        // Find a dominant variant j such that for all i ≠ j: language(i) ⊆ language(j).
        let empty_props = Map::new();
        let dominant_idx = (0..resolved.len()).find(|&j| {
            let (req_j, keys_j) = &info[j];
            let req_j_set: std::collections::BTreeSet<&str> =
                req_j.iter().map(String::as_str).collect();
            let keys_j_set: std::collections::BTreeSet<&str> =
                keys_j.iter().map(String::as_str).collect();
            let props_j = resolved[j]
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or(&empty_props);

            (0..resolved.len()).all(|i| {
                if i == j { return true; }
                let (req_i, keys_i) = &info[i];
                let req_i_set: std::collections::BTreeSet<&str> =
                    req_i.iter().map(String::as_str).collect();
                let keys_i_set: std::collections::BTreeSet<&str> =
                    keys_i.iter().map(String::as_str).collect();
                if !req_j_set.is_subset(&req_i_set) { return false; }
                if !keys_j_set.is_subset(&keys_i_set) { return false; }
                let props_i = resolved[i]
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty_props);
                props_j.iter().all(|(k, sv)| {
                    props_i.get(k).map_or(true, |sv2| sv == sv2)
                })
            })
        });

        let Some(j) = dominant_idx else {
            return Ok(None);
        };

        let dominant = Value::Object(resolved[j].clone());
        Ok(Some(self.convert_schema(&dominant)?))
    }

    /// For closed objects (`additionalProperties: false`), `language(A) ⊆ language(B)` iff:
    ///   1. `B.required ⊆ A.required`  (B requires fewer keys → accepts more objects)
    ///   2. `A.properties.keys() ⊆ B.properties.keys()`  (B allows at least every key A allows)
    ///   3. For every key `k` in A: `B.properties[k] == A.properties[k]`
    ///
    /// When these hold for all `i ≠ j`, `anyOf[A₀…Aₙ] = B` exactly, so compiling just `B`
    /// is language-preserving and eliminates ambiguous duplicate object branches.
    fn try_reduce_anyof_closed_objects(
        &mut self,
        schema: &Map<String, Value>,
        options: &[Value],
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if options.len() < 2 {
            return Ok(None);
        }

        let resolved: Vec<Map<String, Value>> = options
            .iter()
            .map(|opt| {
                if has_structural_keywords(schema) {
                    let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
                    self.merge_resolved_subschemas(&base, std::slice::from_ref(opt))
                } else {
                    self.schema_for_intersection(opt)
                }
            })
            .collect();

        for v in &resolved {
            if v.get("type").and_then(Value::as_str) != Some("object") {
                return Ok(None);
            }
            if !matches!(v.get("additionalProperties"), Some(Value::Bool(false))) {
                return Ok(None);
            }
            for key in &[
                "patternProperties",
                "propertyNames",
                "anyOf",
                "oneOf",
                "allOf",
                "$ref",
                "minProperties",
                "maxProperties",
                "if",
                "then",
                "else",
            ] {
                if v.contains_key(*key) {
                    return Ok(None);
                }
            }
        }

        let info: Vec<(Vec<String>, Vec<String>)> = resolved
            .iter()
            .map(|v| {
                let required: Vec<String> = v
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).collect())
                    .unwrap_or_default();
                let prop_keys: Vec<String> = v
                    .get("properties")
                    .and_then(Value::as_object)
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                (required, prop_keys)
            })
            .collect();

        let empty_props = Map::new();
        let dominant_idx = (0..resolved.len()).find(|&j| {
            let (req_j, keys_j) = &info[j];
            let req_j_set: BTreeSet<&str> = req_j.iter().map(String::as_str).collect();
            let keys_j_set: BTreeSet<&str> = keys_j.iter().map(String::as_str).collect();
            let props_j = resolved[j]
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or(&empty_props);

            (0..resolved.len()).all(|i| {
                if i == j {
                    return true;
                }
                let (req_i, keys_i) = &info[i];
                let req_i_set: BTreeSet<&str> = req_i.iter().map(String::as_str).collect();
                let keys_i_set: BTreeSet<&str> = keys_i.iter().map(String::as_str).collect();
                if !req_j_set.is_subset(&req_i_set) {
                    return false;
                }
                if !keys_i_set.is_subset(&keys_j_set) {
                    return false;
                }
                let props_i = resolved[i]
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty_props);
                props_i.iter().all(|(k, sv)| props_j.get(k) == Some(sv))
            })
        });

        let Some(j) = dominant_idx else {
            return Ok(None);
        };

        let dominant = Value::Object(resolved[j].clone());
        Ok(Some(self.convert_schema(&dominant)?))
    }

    fn convert_structural_branches(
        &mut self,
        schema: &Map<String, Value>,
        keyword: &str,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        let Some(options) = schema.get(keyword).and_then(Value::as_array) else {
            return Ok(None);
        };
        if options.is_empty() {
            return Ok(None);
        }

        if keyword == "oneOf"
            && !self.one_of_is_explicitly_coerced()
            && !self.one_of_options_are_safe_to_coerce(schema, options)
        {
            return Err(GlrMaskError::GrammarParse(
                "oneOf constraints are not supported. Enable 'coerce_one_of' option to approximate oneOf with anyOf".to_string(),
            ));
        }

        if keyword == "oneOf" && self.one_of_is_explicitly_coerced() {
            // Match llguidance's behavior when oneOf coercion is explicitly enabled:
            // compile the oneOf as a plain anyOf-style union of branches.
        }

        if keyword == "anyOf" {
            if let Some(expr) = self.try_reduce_anyof_closed_objects(schema, options)? {
                return Ok(Some(expr));
            }
            // Deliberately keep anyOf lowering rudimentary for now: emit a plain
            // choice over branch object definitions and accept the extra GLR
            // ambiguity instead of trying to factor object unions.
        } else if keyword != "oneOf" {
            return Ok(None);
        }

        // Use the generic branch conversion below for both anyOf and the subset
        // of oneOf cases we are willing to coerce.

        let option_exprs = if has_structural_keywords(schema) {
            let base = Self::schema_without_keys(schema, &["anyOf", "oneOf"]);
            options
                .iter()
                .map(|option| {
                    let merged = Value::Object(
                        self.merge_resolved_subschemas(&base, std::slice::from_ref(option)),
                    );
                    self.convert_schema(&merged)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            self.convert_schema_options(options)?
        };

        Ok(Some(factor_common_affixes(option_exprs)))
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
        let mut suffix_index = 2;
        while self.used_rule_names.contains(&name) {
            name = format!("{base}_{suffix_index}");
            suffix_index += 1;
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
        let expr_result = match self.resolve_local_ref(ref_value).cloned() {
            Ok(target) => match self.convert_schema(&target) {
                Ok(expr) => Ok(expr),
                Err(err) if is_unsat_schema_error(&err) => Ok(never_expr()),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        };
        self.ref_compile_stack.remove(ref_value);
        let expr = expr_result?;
        self.insert_rule(rule_name, expr);
        Ok(())
    }

    fn convert_ref(&mut self, ref_value: &str) -> Result<GrammarExpr, GlrMaskError> {
        let rule_name = self.ensure_ref_rule(ref_value);
        self.compile_ref_rule(ref_value)?;
        Ok(GrammarExpr::Ref(rule_name))
    }

    fn convert_schema(&mut self, schema: &Value) -> Result<GrammarExpr, GlrMaskError> {
        const MAX_CONVERT_DEPTH: usize = 128;
        if self.convert_depth >= MAX_CONVERT_DEPTH {
            return Err(GlrMaskError::GrammarParse(
                "schema conversion exceeded maximum recursion depth (likely circular $ref)".into(),
            ));
        }

        // Schema-level dedup: reuse the expression produced by an earlier
        // identical sub-schema instead of generating duplicate rules.
        // The cache key includes the current draft so that draft-sensitive
        // behavior is preserved.
        let cache_key = format!(
            "{:?}|{}",
            self.current_draft(),
            serde_json::to_string(schema).unwrap_or_default()
        );
        if let Some(cached) = self.schema_convert_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        self.convert_depth += 1;
        let result = self.convert_schema_inner(schema);
        self.convert_depth -= 1;

        if let Ok(ref expr) = result {
            self.schema_convert_cache
                .insert(cache_key, expr.clone());
        }
        result
    }

    fn convert_schema_inner(&mut self, schema: &Value) -> Result<GrammarExpr, GlrMaskError> {
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

            if let Some(expr) = self.convert_structural_branches(object, "anyOf")? {
                return Ok(expr);
            }

            if let Some(expr) = self.convert_structural_branches(object, "oneOf")? {
                return Ok(expr);
            }

            if let Some(all_of) = object.get("allOf").and_then(Value::as_array) {
                if !all_of.is_empty() {
                    let base = Self::schema_without_keys(object, &["allOf"]);
                    let merged = self.merge_resolved_subschemas(&base, all_of);
                    return self.convert_schema(&Value::Object(merged));
                }
            }

            if let Some(type_values) = object.get("type").and_then(Value::as_array) {
                let mut allowed_types: BTreeSet<&str> = type_values
                    .iter()
                    .filter_map(Value::as_str)
                    .collect();
                if allowed_types.contains("number") {
                    allowed_types.remove("integer");
                }
                let options = allowed_types
                    .into_iter()
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
            let remaining_schema = Value::Object(Self::schema_without_keys(object, &["const"]));
            if self.value_satisfies_schema(value, &remaining_schema) {
                return Ok(Some(self.json_literal(value)));
            }
            return Err(unsat_schema_error());
        }

        if let Some(values) = object.get("enum").and_then(Value::as_array) {
            let remaining_schema = Value::Object(Self::schema_without_keys(object, &["enum"]));
            let options: Vec<GrammarExpr> = values
                .iter()
                .filter(|value| self.value_satisfies_schema(value, &remaining_schema))
                .map(|value| self.json_literal(value))
                .collect();
            if !options.is_empty() {
                return Ok(Some(factor_common_affixes(options)));
            }
            return Err(unsat_schema_error());
        }

        Ok(None)
    }

    fn string_value_satisfies_schema(
        &self,
        text: &str,
        schema: &Map<String, Value>,
    ) -> bool {
        if let Some(min_length) = schema.get("minLength").and_then(Value::as_u64) {
            if text.chars().count() < min_length as usize {
                return false;
            }
        }
        if let Some(max_length) = schema.get("maxLength").and_then(Value::as_u64) {
            if text.chars().count() > max_length as usize {
                return false;
            }
        }
        true
    }

    fn object_value_satisfies_schema(
        &self,
        entries: &Map<String, Value>,
        schema: &Map<String, Value>,
    ) -> bool {
        if let Some(min_properties) = schema.get("minProperties").and_then(Value::as_u64) {
            if entries.len() < min_properties as usize {
                return false;
            }
        }
        if let Some(max_properties) = schema.get("maxProperties").and_then(Value::as_u64) {
            if entries.len() > max_properties as usize {
                return false;
            }
        }

        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !entries.contains_key(key) {
                    return false;
                }
            }
        }

        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(properties) = properties {
            for (key, subschema) in properties {
                if let Some(item) = entries.get(key) {
                    if !self.value_satisfies_schema(item, subschema) {
                        return false;
                    }
                }
            }
        }

        match schema.get("additionalProperties") {
            Some(Value::Bool(false)) => {
                if let Some(properties) = properties {
                    if entries.keys().any(|key| !properties.contains_key(key)) {
                        return false;
                    }
                }
            }
            Some(Value::Object(additional_schema)) => {
                for (key, item) in entries {
                    if properties.map(|props| props.contains_key(key)).unwrap_or(false) {
                        continue;
                    }
                    if !self.value_satisfies_schema(item, &Value::Object(additional_schema.clone())) {
                        return false;
                    }
                }
            }
            _ => {}
        }

        true
    }

    fn array_value_satisfies_schema(
        &self,
        items: &[Value],
        schema: &Map<String, Value>,
    ) -> bool {
        let (prefix_items, items_schema, min_items, max_items) =
            self.normalize_array_keywords(schema);
        if items.len() < min_items {
            return false;
        }
        if max_items.map(|max| items.len() > max).unwrap_or(false) {
            return false;
        }

        if !prefix_items.is_empty() {
            for (index, subschema) in prefix_items.iter().enumerate() {
                if let Some(item) = items.get(index) {
                    if !self.value_satisfies_schema(item, subschema) {
                        return false;
                    }
                }
            }
            if matches!(items_schema, Some(Value::Bool(false))) && items.len() > prefix_items.len() {
                return false;
            }
            if let Some(item_schema) = items_schema.as_ref() {
                for item in items.iter().skip(prefix_items.len()) {
                    if !self.value_satisfies_schema(item, item_schema) {
                        return false;
                    }
                }
            }
            return true;
        }

        if matches!(items_schema, Some(Value::Bool(false))) {
            return items.is_empty();
        }
        if let Some(item_schema) = items_schema.as_ref() {
            for item in items {
                if !self.value_satisfies_schema(item, item_schema) {
                    return false;
                }
            }
        }
        true
    }

    fn numeric_value_satisfies_schema(
        &self,
        number: f64,
        schema: &Map<String, Value>,
    ) -> bool {
        let (left, left_inclusive, right, right_inclusive) = normalize_numeric_bounds(schema);
        if left
            .map(|bound| number < bound || (!left_inclusive && number <= bound))
            .unwrap_or(false)
        {
            return false;
        }
        if right
            .map(|bound| number > bound || (!right_inclusive && number >= bound))
            .unwrap_or(false)
        {
            return false;
        }
        if let Some(multiple_of) = schema.get("multipleOf").and_then(Value::as_f64) {
            if multiple_of != 0.0 {
                let quotient = number / multiple_of;
                if (quotient - quotient.round()).abs() > 1e-9 {
                    return false;
                }
            }
        }
        true
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
                return self.value_satisfies_schema(value, target);
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
            if !self.string_value_satisfies_schema(text, object) {
                return false;
            }
        }

        if let Some(entries) = value.as_object() {
            if !self.object_value_satisfies_schema(entries, object) {
                return false;
            }
        }

        if let Some(items) = value.as_array() {
            if !self.array_value_satisfies_schema(items, object) {
                return false;
            }
        }

        if let Some(number) = value.as_f64() {
            if !self.numeric_value_satisfies_schema(number, object) {
                return false;
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
            "number" => self.json_number_type_expr(),
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

    fn build_numeric_ref(&mut self, type_name: &str, schema: &Map<String, Value>) -> GrammarExpr {
        fn integer_bounds_for_number_range(
            left: Option<f64>,
            left_inclusive: bool,
            right: Option<f64>,
            right_inclusive: bool,
        ) -> Option<(Option<i64>, Option<i64>)> {
            let int_left = left.map(|value| {
                if left_inclusive {
                    value.ceil() as i64
                } else {
                    value.floor() as i64 + 1
                }
            });
            let int_right = right.map(|value| {
                if right_inclusive {
                    value.floor() as i64
                } else {
                    value.ceil() as i64 - 1
                }
            });

            if let (Some(int_left), Some(int_right)) = (int_left, int_right) {
                if int_left > int_right {
                    return None;
                }
            }

            Some((int_left, int_right))
        }

        fn number_base_regexes(nonnegative: bool) -> Vec<String> {
            let integer = if nonnegative {
                r#"(0|[1-9][0-9]*)"#
            } else {
                r#"-?(0|[1-9][0-9]*)"#
            };
            let noninteger = if nonnegative {
                JSON_NONNEG_NUMBER_NONINTEGER_REGEX
            } else {
                JSON_NUMBER_NONINTEGER_REGEX
            };
            vec![integer.to_string(), noninteger.to_string()]
        }

        let build_supported_multiple_expr = |
            this: &mut Self,
            regexes: Vec<String>,
            supported: SupportedMultipleOf,
            allow_fractional_zero: bool,
        | {
            let base_expr = compile_regex_union_expr(&regexes);
            let multiple_expr = match supported {
                SupportedMultipleOf::Integer(multiple) => {
                    integer_multiple_expr(multiple, allow_fractional_zero)
                }
                SupportedMultipleOf::ReciprocalPowerOfTen(scale) => {
                    reciprocal_power_of_ten_expr(scale)
                }
            };
            let intersected = LexerExpr::Intersect {
                expr: Box::new(base_expr),
                intersect: Box::new(multiple_expr),
            };
            this.build_lexer_expr(&intersected, "JSON_NUMBER_MULTIPLE_OF")
        };

        let (left, left_inclusive, right, right_inclusive) = normalize_numeric_bounds(schema);
        let multiple_of = schema.get("multipleOf").and_then(Value::as_f64);
        let supported_multiple = multiple_of.and_then(supported_multiple_of);

        let has_bounds = left.is_some() || right.is_some();
        if !has_bounds {
            if let Some(supported) = supported_multiple {
                return match (type_name, supported) {
                    ("integer", SupportedMultipleOf::ReciprocalPowerOfTen(_)) => {
                        self.json_integer_ref()
                    }
                    ("integer", SupportedMultipleOf::Integer(_)) => build_supported_multiple_expr(
                        self,
                        vec![r#"-?(0|[1-9][0-9]*)"#.to_string()],
                        supported,
                        false,
                    ),
                    ("number", _) => build_supported_multiple_expr(
                        self,
                        number_base_regexes(false),
                        supported,
                        true,
                    ),
                    _ => self.json_value_ref(),
                };
            }
            return if type_name == "integer" {
                self.json_integer_ref()
            } else {
                self.json_number_type_expr()
            };
        }

        // Only the exact lower bound of 0 can safely use the generic non-negative rules.
        let use_nonneg_shortcut = right.is_none() && left == Some(0.0) && left_inclusive;
        if use_nonneg_shortcut {
            if let Some(supported) = supported_multiple {
                return match (type_name, supported) {
                    ("integer", SupportedMultipleOf::ReciprocalPowerOfTen(_)) => {
                        GrammarExpr::Ref(JSON_NONNEG_INTEGER_RULE.into())
                    }
                    ("integer", SupportedMultipleOf::Integer(_)) => build_supported_multiple_expr(
                        self,
                        vec![r#"(0|[1-9][0-9]*)"#.to_string()],
                        supported,
                        false,
                    ),
                    ("number", _) => build_supported_multiple_expr(
                        self,
                        number_base_regexes(true),
                        supported,
                        true,
                    ),
                    _ => self.json_value_ref(),
                };
            }
            return if type_name == "integer" {
                GrammarExpr::Ref(JSON_NONNEG_INTEGER_RULE.into())
            } else {
                choice_or_single(vec![
                    GrammarExpr::Ref(JSON_NONNEG_INTEGER_RULE.into()),
                    GrammarExpr::Ref(JSON_NONNEG_NUMBER_RULE.into()),
                ])
            };
        }

        // Build precise range regex
        use crate::import::numeric_range::{
            rx_float_range,
            rx_int_range,
            rx_noninteger_float_range,
        };

        if type_name == "integer" {
            let int_left = left.map(|l| if left_inclusive { l as i64 } else { l as i64 + 1 });
            let int_right = right.map(|r| if right_inclusive { r as i64 } else { r as i64 - 1 });
            return match rx_int_range(int_left, int_right) {
                Ok(regex) => {
                    if let Some(supported) = supported_multiple {
                        match supported {
                            SupportedMultipleOf::ReciprocalPowerOfTen(_) => GrammarExpr::RawRegex(regex),
                            SupportedMultipleOf::Integer(_) => build_supported_multiple_expr(
                                self,
                                vec![regex],
                                supported,
                                false,
                            ),
                        }
                    } else {
                        GrammarExpr::RawRegex(regex)
                    }
                }
                Err(_) => self.json_integer_ref(),
            };
        }

        let float_regex_result = rx_float_range(left, right, left_inclusive, right_inclusive);
        let non_integer_regex_result =
            rx_noninteger_float_range(left, right, left_inclusive, right_inclusive);
        let integer_range_result = integer_bounds_for_number_range(
            left,
            left_inclusive,
            right,
            right_inclusive,
        )
        .map(|(int_left, int_right)| rx_int_range(int_left, int_right));

        match (float_regex_result, non_integer_regex_result, integer_range_result) {
            (Ok(_float_regex), Ok(Some(non_integer_regex)), Some(Ok(int_regex))) => {
                if let Some(supported) = supported_multiple {
                    build_supported_multiple_expr(
                        self,
                        vec![int_regex, non_integer_regex],
                        supported,
                        true,
                    )
                } else {
                    choice_or_single(vec![
                        regex_expr(int_regex),
                        regex_expr(non_integer_regex),
                    ])
                }
            }
            (Ok(_float_regex), Ok(Some(non_integer_regex)), None) => {
                if let Some(supported) = supported_multiple {
                    build_supported_multiple_expr(
                        self,
                        vec![non_integer_regex],
                        supported,
                        true,
                    )
                } else {
                    regex_expr(non_integer_regex)
                }
            }
            (Ok(_float_regex), Ok(None), Some(Ok(int_regex))) => {
                if let Some(supported) = supported_multiple {
                    build_supported_multiple_expr(
                        self,
                        vec![int_regex],
                        supported,
                        true,
                    )
                } else {
                    regex_expr(int_regex)
                }
            }
            _ => self.json_number_type_expr(),
        }
    }

    fn build_string_expr(&mut self, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        let cap = max_string_length_cap();
        let min_len = {
            let raw = schema
                .get("minLength")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(0);
            match cap {
                // If raw exceeds cap, remove the constraint entirely (→ 0).
                Some(c) if raw > c => 0,
                _ => raw,
            }
        };
        let max_len = {
            let raw = schema
                .get("maxLength")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            match (cap, raw) {
                // If maxLength exceeds cap, remove it entirely (→ None/unbounded).
                (Some(c), Some(ml)) if ml > c => None,
                _ => raw,
            }
        };

        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            let Some(pattern) = prune_pattern_branches_for_min_length(pattern, min_len) else {
                return Ok(self.extract_terminal_rule(never_expr(), "JSON_STRING_PATTERN_UNSAT"));
            };
            if let Some(unit_pattern) = simple_repeated_single_char_pattern(&pattern) {
                if min_len > 0 || max_len.is_some() {
                    return Ok(self.build_bounded_string_from_unit_regex(&unit_pattern, min_len, max_len));
                }
                return Ok(self.build_json_wrapped_fullmatch_pattern(
                    &format!("{}+", unit_pattern),
                    "JSON_STRING_PATTERN_FULLMATCH",
                ));
            }
            // Fixed-length unanchored pattern optimization:
            // When an unanchored pattern has a fixed byte length *and* the string's
            // minLength/maxLength are both equal to that length, the search semantics
            // (STRING_CHAR* pattern STRING_CHAR*) collapse to fullmatch — there is no
            // room for any padding bytes.  This eliminates unbounded-length padding
            // that would otherwise cause DFA state explosion.
            if !pattern_all_branches_anchored(&pattern) {
                let expr = parse_regex(&pattern, true);
                let (pat_min_bytes, pat_max_bytes) = regex_byte_length_bounds(&expr);
                if pat_min_bytes > 0 && pat_max_bytes == Some(pat_min_bytes) {
                    let pat_len = pat_min_bytes;
                    if min_len == pat_len && max_len == Some(pat_len)
                    {
                        return Ok(self.build_json_wrapped_fullmatch_pattern(
                            &pattern,
                            "JSON_STRING_PATTERN_FULLMATCH",
                        ));
                    }
                }
            }
            if pattern_all_branches_anchored(&pattern) {
                // Every branch is ^…$, so json_wrapped_pattern produces no
                // <string_tail> padding — safe from DFA explosion.
                // Check whether the pruned pattern's minimum match length
                // already satisfies minLength.  If so, the pattern alone is
                // sufficient.  Otherwise, intersect with a length regex to
                // enforce minLength (e.g. pattern ^(.*)$ can match "").
                let pattern_min = pattern_min_char_count(&pattern).unwrap_or(0);
                if pattern_min >= min_len {
                    return Ok(self.build_json_wrapped_pattern(&pattern, "JSON_STRING_PATTERN"));
                }
                // Pattern can produce strings shorter than minLength.
                // Intersect with a minimum-length regex.  Use {min,} to
                // keep the length DFA small (avoids explosion from large
                // maxLength values).
                let search_regex = string_value_body_regex(&json_search_pattern(&pattern));
                let length_inner = match max_len {
                    Some(ml) if ml <= 100 => format!(r#"(?:{}){{{},{}}}"#, JSON_STRING_CHAR_PATTERN, min_len, ml),
                    _ => format!(r#"(?:{}){{{},}}"#, JSON_STRING_CHAR_PATTERN, min_len),
                };
                let length_regex = string_value_body_regex(&length_inner);
                let intersected = LexerExpr::Intersect {
                    expr: Box::new(parse_regex(&search_regex, true)),
                    intersect: Box::new(parse_regex(&length_regex, true)),
                };
                let body = self.build_lexer_expr(&intersected, "JSON_STRING_PATTERN_ANCHORED_BOUNDED");
                return Ok(wrap_string_value_terminal(body));
            }
            if min_len > 0 || max_len.is_some() {
                // Unanchored pattern with length bounds. For reasonably small
                // maxLength values, build the exact intersection of:
                // 1. the quoted JSON-string search-pattern regex, and
                // 2. the quoted JSON-string length-bounded regex.
                // This preserves JSON Schema search semantics while enforcing
                // minLength/maxLength exactly.
                const MAX_BOUNDED_SEARCH_TAIL: usize = 100;
                if let Some(ml) = max_len {
                    if ml <= MAX_BOUNDED_SEARCH_TAIL {
                        let search_regex = string_value_body_regex(&json_search_pattern(&pattern));
                        let length_regex = json_wrapped_string_length_regex(min_len, ml);
                        let intersected = LexerExpr::Intersect {
                            expr: Box::new(parse_regex(&search_regex, true)),
                            intersect: Box::new(parse_regex(&length_regex, true)),
                        };
                        let body = self.build_lexer_expr(&intersected, "JSON_STRING_PATTERN_BOUNDED");
                        return Ok(wrap_string_value_terminal(body));
                    }
                }
                // maxLength is too large for exact intersection — apply pattern
                // with minLength enforcement only (drop maxLength to avoid
                // DFA explosion).
                if min_len > 0 {
                    let search_regex = string_value_body_regex(&json_search_pattern(&pattern));
                    let length_inner = format!(r#"(?:{}){{{},}}"#, JSON_STRING_CHAR_PATTERN, min_len);
                    let length_regex = string_value_body_regex(&length_inner);
                    let intersected = LexerExpr::Intersect {
                        expr: Box::new(parse_regex(&search_regex, true)),
                        intersect: Box::new(parse_regex(&length_regex, true)),
                    };
                    let body = self.build_lexer_expr(&intersected, "JSON_STRING_PATTERN_MINLEN");
                    return Ok(wrap_string_value_terminal(body));
                }
                return Ok(self.build_json_wrapped_pattern(&pattern, "JSON_STRING_PATTERN"));
            } else {
                return Ok(self.build_json_wrapped_pattern(&pattern, "JSON_STRING_PATTERN"));
            }
        }

        if let Some(format_name) = schema.get("format").and_then(Value::as_str) {
            // Email regexes currently leak byte-level lexical structure into
            // parser-visible terminals. Keep them on the regular string path.
            if format_name != "email" {
                return self.build_format_string_expr(format_name);
            }
        }
        if min_len == 0 && max_len.is_none() {
            return Ok(self.json_string_ref());
        }

        if self.should_split_bounded_string(min_len, max_len) {
            let open = split_open_quote();
            let close = split_close_quote();
            // When min_len > 0 the exact-part terminal must start with the
            // opening quote so the tokenizer DFA cannot conflate it with
            // JSON_STRING_BODY (which also follows a standalone '"' terminal).
            let actually_split_open = open && min_len == 0;
            let prefix = if actually_split_open { None } else { Some(literal_expr(b"\"")) };
            let suffix = if close { None } else { Some(literal_expr(b"\"")) };
            let body = self.build_split_json_string_body_wrapped(min_len, max_len, prefix, suffix);

            let mut result_parts = Vec::new();
            if actually_split_open {
                result_parts.push(literal_expr(b"\""));
            }
            result_parts.push(body);
            if close {
                result_parts.push(literal_expr(b"\""));
            }

            return Ok(self.extract_rule(
                sequence_or_single(result_parts),
                "json_string_bounded_split",
            ));
        }

        let bounded_body = match max_len {
            Some(0) if min_len == 0 => empty_expr(),
            Some(max_len) => GrammarExpr::RepeatRange {
                expr: Box::new(self.json_string_char_ref()),
                min: min_len,
                max: max_len,
            },
            None => {
                let mut parts = Vec::new();
                if min_len > 0 {
                    parts.push(GrammarExpr::RepeatRange {
                        expr: Box::new(self.json_string_char_ref()),
                        min: min_len,
                        max: min_len,
                    });
                }
                parts.push(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
                sequence_or_single(parts)
            }
        };

        let (terminal_body, wrap) = wrap_string_value_expr_parts(bounded_body);
        let body = self.extract_terminal_rule(terminal_body, "JSON_STRING_BOUNDED");
        Ok(wrap(body))
    }

    fn build_bounded_string_from_unit_regex(
        &mut self,
        unit_pattern: &str,
        min_len: usize,
        max_len: Option<usize>,
    ) -> GrammarExpr {
        let unit_expr = self.extract_terminal_rule(
            parsed_regex_expr(unit_pattern, true),
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

        let (terminal_body, wrap) = wrap_string_value_expr_parts(bounded_body);
        let body = self.extract_terminal_rule(terminal_body, "JSON_STRING_BOUNDED_PATTERN");
        wrap(body)
    }

    fn build_format_string_expr(&mut self, format_name: &str) -> Result<GrammarExpr, GlrMaskError> {
        match format_name {
            "date" | "time" | "date-time" => {
                let body_inner = match format_name {
                    "date" => json_date_body_expr(),
                    "time" => json_time_body_expr(),
                    _ => json_date_time_body_expr(),
                };
                let (terminal_body, wrap) = wrap_string_value_expr_parts(body_inner);
                let body = self.extract_terminal_rule(terminal_body, "JSON_FORMAT_STRING");
                Ok(wrap(body))
            }
            "hostname" => {
                let label = self.extract_terminal_rule(
                    parsed_regex_expr(json_hostname_label_pattern(), true),
                    "JSON_FORMAT_HOSTNAME_LABEL",
                );
                Ok(quoted_expr(sequence_or_single(vec![
                    label.clone(),
                    GrammarExpr::Repeat(Box::new(sequence_or_single(vec![
                        literal_expr(b"."),
                        label,
                    ]))),
                ])))
            }
            "uri" if use_structured_uri() => {
                Ok(self.build_llguidance_uri_expr())
            }
            _ => json_format_pattern(format_name)
                .map(|pattern| self.build_json_wrapped_fullmatch_pattern(&pattern, "JSON_FORMAT_STRING"))
                .ok_or_else(|| GlrMaskError::GrammarParse(format!("Unknown format: {format_name}"))),
        }
    }

    fn build_llguidance_uri_expr(&mut self) -> GrammarExpr {
        let pattern = concat!(
            r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+\-.]*)",
            r":",
            r"(?:",
            r"//",
            r"(?:",
            r"(?P<userinfo>(?:[a-zA-Z0-9\-._~!$&'()*+,;=:]|%[0-9a-fA-F]{2})*)",
            r"@",
            r")?",
            r"(?P<host>",
            r"\[",
            r"(?:",
            r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|",
            r"(?:[0-9a-fA-F]{1,4}:){1,7}:|",
            r"(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|",
            r"(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|",
            r"(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|",
            r"(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|",
            r"(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|",
            r"[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}|",
            r":(?::[0-9a-fA-F]{1,4}){1,7}|",
            r"::|",
            r"v[0-9a-fA-F]+\.[a-zA-Z0-9\-._~!$&'()*+,;=:]+",
            r")",
            r"\]|",
            r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])|",
            r"(?:[a-zA-Z0-9\-._~!$&'()*+,;=]|%[0-9a-fA-F]{2})*",
            r")",
            r"(?::(?P<port>[0-9]*))?",
            r"(?P<path_abempty>(?:/(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)",
            r"|",
            r"(?P<path_absolute>/(?:(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})+(?:/(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)?)",
            r"|",
            r"(?P<path_rootless>(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})+(?:/(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@]|%[0-9a-fA-F]{2})*)*)",
            r"|",
            r"(?P<path_empty>)",
            r")",
            r"(?:\?(?P<query>(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]|%[0-9a-fA-F]{2})*))?",
            r"(?:\#(?P<fragment>(?:[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]|%[0-9a-fA-F]{2})*))?"
        );
        self.build_json_wrapped_fullmatch_pattern(pattern, "JSON_FORMAT_URI")
    }

    fn build_structured_uri_expr(&mut self) -> GrammarExpr {
        // Ablation switch: set URI_ABLATE=<csv> to disable URI parts for
        // perf investigation. Parts: ipv6, ipv_future, ip_literal, ipv4,
        // reg_name, userinfo, port, path_abempty, path_absolute, path_rootless,
        // hier_part_extra (drops the 3 non-authority hier-part branches),
        // query, fragment, scheme_rich (collapses scheme body).
        let ablate: std::collections::BTreeSet<String> = std::env::var("URI_ABLATE")
            .ok()
            .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();
        let ablated = |name: &str| ablate.contains(name);
        let run_chunk_max = uri_run_chunk_max();

        let uri_hexdig = self.insert_named_terminal_rule(
            "URI_HEXDIG",
            GrammarExpr::CharClass {
                def: "0-9a-fA-F".into(),
                negate: false,
                utf8: true,
            },
        );
        let uri_alpha_terminal = self.insert_named_terminal_rule(
            "URI_ALPHA",
            GrammarExpr::CharClass {
                def: "a-zA-Z".into(),
                negate: false,
                utf8: true,
            },
        );
        let uri_scheme_char_terminal = self.insert_named_terminal_rule(
            "URI_SCHEME_CHAR",
            regex_expr(&uri_charclass_run_regex("a-zA-Z0-9+\\-.", run_chunk_max)),
        );
        let uri_alpha = self.insert_uri_rule("uri_alpha_char", uri_alpha_terminal.clone());
        let uri_scheme_char =
            self.insert_uri_rule("uri_scheme_char", uri_scheme_char_terminal.clone());
        let uri_userinfo = self.insert_named_terminal_rule(
            "URI_USERINFO",
            regex_expr(r#"(?:[a-zA-Z0-9\-._~!$&'()*+,;=:]|%[0-9A-Fa-f]{2})*@"#),
        );
        let uri_ipvfuture_char = self.insert_named_terminal_rule(
            "URI_IPVFUTURE_CHAR",
            GrammarExpr::CharClass {
                def: r"a-zA-Z0-9\-._~!$&'()*+,;=:".into(),
                negate: false,
                utf8: true,
            },
        );
        let uri_dec_octet = self.insert_named_terminal_rule(
            "URI_DEC_OCTET",
            regex_expr(r#"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"#),
        );
        let uri_reg_name_char_terminal = self.insert_named_terminal_rule(
            "URI_REG_NAME_CHAR",
            regex_expr(&uri_charclass_run_regex(r"a-zA-Z0-9\-._~!$&'()*+,;=", run_chunk_max)),
        );
        let uri_port_digit = self.insert_named_terminal_rule(
            "URI_PORT",
            regex_expr(&uri_charclass_run_regex("0-9", run_chunk_max)),
        );
        let uri_pchar_base_terminal = self.insert_named_terminal_rule(
            "URI_PCHAR",
            regex_expr(&uri_charclass_run_regex(r"a-zA-Z0-9\-._~!$&'()*+,;=:@", run_chunk_max)),
        );
        let uri_query_frag_char_terminal = self.insert_named_terminal_rule(
            "URI_QUERY_FRAG_CHAR",
            regex_expr(&uri_charclass_run_regex(r"a-zA-Z0-9\-._~!$&'()*+,;=:@/?", run_chunk_max)),
        );
        let uri_pchar_base =
            self.insert_uri_rule("uri_pchar_char", uri_pchar_base_terminal.clone());
        let uri_query_frag_char =
            self.insert_uri_rule("uri_query_frag_char", uri_query_frag_char_terminal.clone());
        let uri_reg_name_char =
            self.insert_uri_rule("uri_reg_name_char", uri_reg_name_char_terminal.clone());

        let uri_pct_encoded = self.insert_uri_rule(
            "uri_pct_encoded",
            sequence_or_single(vec![
                literal_expr(b"%"),
                GrammarExpr::RepeatRange {
                    expr: Box::new(uri_hexdig.clone()),
                    min: 2,
                    max: 2,
                },
            ]),
        );
        let uri_pchar = self.insert_uri_rule(
            "uri_pchar",
            choice_or_single(vec![uri_pchar_base.clone(), uri_pct_encoded.clone()]),
        );
        let uri_query_frag = self.insert_uri_rule(
            "uri_query_frag",
            choice_or_single(vec![uri_query_frag_char.clone(), uri_pct_encoded.clone()]),
        );
        let uri_h16_colon = self.insert_uri_rule(
            "uri_h16_colon",
            sequence_or_single(vec![
                GrammarExpr::RepeatRange {
                    expr: Box::new(uri_hexdig.clone()),
                    min: 1,
                    max: 4,
                },
                literal_expr(b":"),
            ]),
        );
        let uri_colon_h16 = self.insert_uri_rule(
            "uri_colon_h16",
            sequence_or_single(vec![
                literal_expr(b":"),
                GrammarExpr::RepeatRange {
                    expr: Box::new(uri_hexdig.clone()),
                    min: 1,
                    max: 4,
                },
            ]),
        );

        let uri_ipv6_address_is_terminal =
            uri_rule_should_be_terminal("uri_ipv6_address").unwrap_or(false);
        let uri_ipv6_address_expr = if ablated("ipv6") {
            literal_expr(b"::")
        } else {
            let ipv6_alts = vec![
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 7,
                        max: 7,
                    },
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_hexdig.clone()),
                        min: 1,
                        max: 4,
                    },
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 1,
                        max: 7,
                    },
                    literal_expr(b":"),
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 1,
                        max: 6,
                    },
                    literal_expr(b":"),
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_hexdig.clone()),
                        min: 1,
                        max: 4,
                    },
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 1,
                        max: 5,
                    },
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_colon_h16.clone()),
                        min: 1,
                        max: 2,
                    },
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 1,
                        max: 4,
                    },
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_colon_h16.clone()),
                        min: 1,
                        max: 3,
                    },
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 1,
                        max: 3,
                    },
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_colon_h16.clone()),
                        min: 1,
                        max: 4,
                    },
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_h16_colon.clone()),
                        min: 1,
                        max: 2,
                    },
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_colon_h16.clone()),
                        min: 1,
                        max: 5,
                    },
                ]),
                sequence_or_single(vec![
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_hexdig.clone()),
                        min: 1,
                        max: 4,
                    },
                    literal_expr(b":"),
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_colon_h16.clone()),
                        min: 1,
                        max: 6,
                    },
                ]),
                sequence_or_single(vec![
                    literal_expr(b":"),
                    GrammarExpr::RepeatRange {
                        expr: Box::new(uri_colon_h16.clone()),
                        min: 1,
                        max: 7,
                    },
                ]),
                literal_expr(b"::"),
            ];
            if uri_ipv6_alt_nonterminals() && !uri_ipv6_address_is_terminal {
                let alt_refs = ipv6_alts
                    .into_iter()
                    .enumerate()
                    .map(|(index, expr)| {
                        let name = format!("uri_ipv6_alt_{}", index + 1);
                        self.insert_rule(name.clone(), expr);
                        GrammarExpr::Ref(name)
                    })
                    .collect::<Vec<_>>();
                choice_or_single(alt_refs)
            } else {
                choice_or_single(ipv6_alts)
            }
        };
        let uri_ipv6_address = self.insert_uri_rule("uri_ipv6_address", uri_ipv6_address_expr);

        let uri_scheme = self.insert_uri_rule(
            "uri_scheme",
            if ablated("scheme_rich") {
                uri_alpha.clone()
            } else {
                sequence_or_single(vec![
                    uri_alpha.clone(),
                    GrammarExpr::Repeat(Box::new(uri_scheme_char.clone())),
                ])
            },
        );
        let uri_query = self.insert_uri_rule(
            "uri_query",
            sequence_or_single(vec![
                literal_expr(b"?"),
                GrammarExpr::Repeat(Box::new(uri_query_frag.clone())),
            ]),
        );
        let uri_fragment = self.insert_uri_rule(
            "uri_fragment",
            sequence_or_single(vec![
                literal_expr(b"#"),
                GrammarExpr::Repeat(Box::new(uri_query_frag.clone())),
            ]),
        );

        self.insert_rule(
            "uri_ipv_future",
            sequence_or_single(vec![
                literal_expr(b"v"),
                GrammarExpr::RepeatOne(Box::new(uri_hexdig.clone())),
                literal_expr(b"."),
                GrammarExpr::RepeatOne(Box::new(uri_ipvfuture_char.clone())),
            ]),
        );
        let uri_ipv_future = GrammarExpr::Ref("uri_ipv_future".into());

        self.insert_rule(
            "uri_ipv4_address",
            sequence_or_single(vec![
                uri_dec_octet.clone(),
                literal_expr(b"."),
                uri_dec_octet.clone(),
                literal_expr(b"."),
                uri_dec_octet.clone(),
                literal_expr(b"."),
                uri_dec_octet.clone(),
            ]),
        );
        let uri_ipv4_address = GrammarExpr::Ref("uri_ipv4_address".into());

        self.insert_rule(
            "uri_reg_name",
            GrammarExpr::Repeat(Box::new(choice_or_single(vec![
                uri_reg_name_char.clone(),
                uri_pct_encoded.clone(),
            ]))),
        );
        let uri_reg_name = GrammarExpr::Ref("uri_reg_name".into());

        let ip_literal_choice = if ablated("ipv_future") {
            uri_ipv6_address.clone()
        } else {
            choice_or_single(vec![uri_ipv6_address.clone(), uri_ipv_future.clone()])
        };
        self.insert_rule(
            "uri_ip_literal",
            sequence_or_single(vec![
                literal_expr(b"["),
                ip_literal_choice,
                literal_expr(b"]"),
            ]),
        );
        let uri_ip_literal = GrammarExpr::Ref("uri_ip_literal".into());

        self.insert_rule(
            "uri_host",
            if ablated("reg_name") && ablated("ip_literal") && ablated("ipv4") {
                uri_alpha.clone()
            } else {
                let mut host_alts = Vec::new();
                if !ablated("ip_literal") {
                    host_alts.push(uri_ip_literal.clone());
                }
                if !ablated("ipv4") {
                    host_alts.push(uri_ipv4_address.clone());
                }
                if !ablated("reg_name") {
                    host_alts.push(uri_reg_name.clone());
                }
                choice_or_single(host_alts)
            },
        );
        let uri_host = GrammarExpr::Ref("uri_host".into());

        self.insert_rule(
            "uri_port",
            sequence_or_single(vec![
                literal_expr(b":"),
                GrammarExpr::Repeat(Box::new(uri_port_digit.clone())),
            ]),
        );
        let uri_port = GrammarExpr::Ref("uri_port".into());

        self.insert_rule(
            "uri_authority",
            sequence_or_single(vec![
                if ablated("userinfo") {
                    empty_expr()
                } else {
                    GrammarExpr::Optional(Box::new(uri_userinfo.clone()))
                },
                uri_host.clone(),
                if ablated("port") {
                    empty_expr()
                } else {
                    GrammarExpr::Optional(Box::new(uri_port.clone()))
                },
            ]),
        );
        let uri_authority = GrammarExpr::Ref("uri_authority".into());

        self.insert_rule(
            "uri_path_abempty",
            GrammarExpr::Repeat(Box::new(sequence_or_single(vec![
                literal_expr(b"/"),
                GrammarExpr::Repeat(Box::new(uri_pchar.clone())),
            ]))),
        );
        let uri_path_abempty = GrammarExpr::Ref("uri_path_abempty".into());

        self.insert_rule(
            "uri_path_absolute",
            sequence_or_single(vec![
                literal_expr(b"/"),
                GrammarExpr::Optional(Box::new(sequence_or_single(vec![
                    GrammarExpr::RepeatOne(Box::new(uri_pchar.clone())),
                    GrammarExpr::Repeat(Box::new(sequence_or_single(vec![
                        literal_expr(b"/"),
                        GrammarExpr::Repeat(Box::new(uri_pchar.clone())),
                    ]))),
                ]))),
            ]),
        );
        let uri_path_absolute = GrammarExpr::Ref("uri_path_absolute".into());

        self.insert_rule(
            "uri_path_rootless",
            sequence_or_single(vec![
                GrammarExpr::RepeatOne(Box::new(uri_pchar.clone())),
                GrammarExpr::Repeat(Box::new(sequence_or_single(vec![
                    literal_expr(b"/"),
                    GrammarExpr::Repeat(Box::new(uri_pchar.clone())),
                ]))),
            ]),
        );
        let uri_path_rootless = GrammarExpr::Ref("uri_path_rootless".into());

        let mut hier_alts = vec![sequence_or_single(vec![
            literal_expr(b"//"),
            uri_authority,
            if ablated("path_abempty") {
                empty_expr()
            } else {
                uri_path_abempty.clone()
            },
        ])];
        if !ablated("hier_part_extra") {
            if !ablated("path_absolute") {
                hier_alts.push(uri_path_absolute);
            }
            if !ablated("path_rootless") {
                hier_alts.push(uri_path_rootless);
            }
            hier_alts.push(empty_expr());
        }
        self.insert_rule("uri_hier_part", choice_or_single(hier_alts));
        let uri_hier_part = GrammarExpr::Ref("uri_hier_part".into());

        self.insert_rule(
            "uri",
            sequence_or_single(vec![
                uri_scheme,
                literal_expr(b":"),
                uri_hier_part,
                if ablated("query") {
                    empty_expr()
                } else {
                    GrammarExpr::Optional(Box::new(uri_query))
                },
                if ablated("fragment") {
                    empty_expr()
                } else {
                    GrammarExpr::Optional(Box::new(uri_fragment))
                },
            ]),
        );

        quoted_expr(GrammarExpr::Ref("uri".into()))
    }

    fn json_literal(&self, value: &Value) -> GrammarExpr {
        json_value_literal_expr(value)
    }

    fn json_key_colon_literal(&self, text: &str) -> GrammarExpr {
        let body = literal_expr(&key_colon_literal_body_bytes(text));
        wrap_key_colon_terminal(body)
    }

    fn fused_json_key_colon_literal(&self, text: &str) -> GrammarExpr {
        let mut bytes = json_string_literal_bytes(text);
        bytes.extend_from_slice(b": ");
        literal_expr(&bytes)
    }

    fn build_merged_literal_key_value_expr(
        &self,
        leading_literal: &[u8],
        key: &str,
        value_expr: GrammarExpr,
    ) -> GrammarExpr {
        let key_expr = self.json_key_colon_literal(key);
        let mut parts = match key_expr {
            GrammarExpr::Sequence(parts) => parts,
            other => vec![other],
        };

        if let Some(GrammarExpr::Literal(first)) = parts.first_mut() {
            let mut merged = Vec::with_capacity(leading_literal.len() + first.len());
            merged.extend_from_slice(leading_literal);
            merged.extend_from_slice(first);
            *first = merged;
        } else {
            parts.insert(0, literal_expr(leading_literal));
        }

        let key_expr = sequence_or_single(parts);
        let (mut parts, value_expr) = if let Some((first, rest)) = self.try_take_leading_container_delim_expr(value_expr.clone()) {
            if let Some(fused_key_expr) = self.try_append_suffix_to_trailing_literal_expr(key_expr.clone(), &[first]) {
                (
                    match fused_key_expr {
                        GrammarExpr::Sequence(inner) => inner,
                        other => vec![other],
                    },
                    rest,
                )
            } else {
                (
                    match key_expr {
                        GrammarExpr::Sequence(inner) => inner,
                        other => vec![other],
                    },
                    value_expr,
                )
            }
        } else {
            (
                match key_expr {
                    GrammarExpr::Sequence(inner) => inner,
                    other => vec![other],
                },
                value_expr,
            )
        };

        // Flatten the value into parts to avoid nested Sequences
        match value_expr {
            GrammarExpr::Sequence(inner) => parts.extend(inner),
            other => parts.push(other),
        }
        sequence_or_single(parts)
    }

    fn build_fused_merged_literal_key_value_expr(
        &self,
        leading_literal: &[u8],
        key: &str,
        value_expr: GrammarExpr,
    ) -> GrammarExpr {
        let key_expr = self.fused_json_key_colon_literal(key);
        let mut parts = match key_expr {
            GrammarExpr::Sequence(parts) => parts,
            other => vec![other],
        };

        if let Some(GrammarExpr::Literal(first)) = parts.first_mut() {
            let mut merged = Vec::with_capacity(leading_literal.len() + first.len());
            merged.extend_from_slice(leading_literal);
            merged.extend_from_slice(first);
            *first = merged;
        } else {
            parts.insert(0, literal_expr(leading_literal));
        }

        let key_expr = sequence_or_single(parts);
        let (mut parts, value_expr) = if let Some((first, rest)) = self.try_take_leading_container_delim_expr(value_expr.clone()) {
            if let Some(fused_key_expr) = self.try_append_suffix_to_trailing_literal_expr(key_expr.clone(), &[first]) {
                (
                    match fused_key_expr {
                        GrammarExpr::Sequence(inner) => inner,
                        other => vec![other],
                    },
                    rest,
                )
            } else {
                (
                    match key_expr {
                        GrammarExpr::Sequence(inner) => inner,
                        other => vec![other],
                    },
                    value_expr,
                )
            }
        } else {
            (
                match key_expr {
                    GrammarExpr::Sequence(inner) => inner,
                    other => vec![other],
                },
                value_expr,
            )
        };

        match value_expr {
            GrammarExpr::Sequence(inner) => parts.extend(inner),
            other => parts.push(other),
        }
        sequence_or_single(parts)
    }

    fn build_required_ordered_object_body_expr(
        &self,
        ordered: &[(String, GrammarExpr, bool)],
        trailing_literal: Option<&[u8]>,
    ) -> GrammarExpr {
        let mut parts = vec![literal_expr(b"{")];

        for (index, (key, value_expr, _)) in ordered.iter().enumerate() {
            let mut leading_literal = if index == 0 {
                Vec::new()
            } else {
                JSON_ITEM_SEPARATOR.to_vec()
            };
            if index > 0 {
                if !leading_literal.is_empty() {
                    if let Some(prev_expr) = parts.pop() {
                        if let Some(fused_prev) = self.try_append_suffix_to_trailing_literal_expr(prev_expr.clone(), &leading_literal) {
                            parts.push(fused_prev);
                            leading_literal.clear();
                        } else {
                            parts.push(prev_expr);
                        }
                    }
                }
            }
            parts.push(self.build_merged_literal_key_value_expr(
                &leading_literal,
                key,
                value_expr.clone(),
            ));
        }

        if let Some(trailing_literal) = trailing_literal {
            if let Some(last_expr) = parts.pop() {
                if let Some(fused_last) = self.try_append_suffix_to_trailing_literal_expr(last_expr.clone(), trailing_literal) {
                    parts.push(fused_last);
                    return sequence_or_single(parts);
                } else {
                    parts.push(last_expr);
                }
            }

            parts.push(literal_expr(trailing_literal));
        }

        sequence_or_single(parts)
    }

    fn build_closed_required_ordered_object_expr(
        &self,
        ordered: &[(String, GrammarExpr, bool)],
    ) -> GrammarExpr {
        maybe_fuse_finite_literal_expr(
            self.build_required_ordered_object_body_expr(ordered, Some(b"}")),
            &format!("closed_required_ordered_object:items={}", ordered.len()),
        )
    }

    fn build_ordered_object_body_separated_sequence_expr(
        &self,
        ordered: &[(String, GrammarExpr, bool)],
        allow_empty: bool,
    ) -> (GrammarExpr, bool) {
        if ordered.is_empty() {
            return (empty_expr(), allow_empty);
        }

        if ordered.len() == 1 {
            let (key, value_expr, is_required) = &ordered[0];
            let item = self.build_merged_literal_key_value_expr(b"", key, value_expr.clone());
            return if *is_required {
                (item, false)
            } else {
                // Keep the tree item itself non-nullable for readability; the
                // caller handles empty-object/body variants when `can_be_empty`
                // is true.
                (item, allow_empty)
            };
        }

        let items = ordered
            .iter()
            .map(|(key, value_expr, is_required)| {
                (
                    self.build_merged_literal_key_value_expr(b"", key, value_expr.clone()),
                    *is_required,
                )
            })
            .collect::<Vec<_>>();

        let can_be_empty = items.iter().all(|(_, is_required)| !*is_required);
        (
            GrammarExpr::SeparatedSequence {
                items,
                separator: Box::new(self.json_item_separator_expr()),
                allow_empty,
            },
            allow_empty && can_be_empty,
        )
    }

    fn json_item_separator_expr(&self) -> GrammarExpr {
        if split_item_separator() {
            sequence_or_single(vec![literal_expr(b","), literal_expr(b" ")])
        } else {
            // Default: keep as a single fused literal token.
            literal_expr(JSON_ITEM_SEPARATOR)
        }
    }

    fn normalized_additional_properties_schema(
        &mut self,
        additional_properties: Option<&Value>,
    ) -> Option<Value> {
        if no_additional_properties() {
            return None;
        }

        let schema = match additional_properties {
            Some(Value::Bool(false)) => None,
            Some(Value::Object(map)) => Some(Value::Object(map.clone())),
            // Absent or `true`: use default based on env var
            _ => {
                if additional_properties.is_none() && additional_properties_default_false() {
                    None
                } else {
                    Some(serde_json::json!({}))
                }
            }
        };

        if schema
            .as_ref()
            .map(|schema| self.is_certainly_unsatisfiable(schema))
            .unwrap_or(false)
        {
            None
        } else {
            schema
        }
    }

    fn pattern_property_entries(
        pattern_properties: Option<&Map<String, Value>>,
    ) -> Vec<(String, Value)> {
        pattern_properties
            .map(|patterns| {
                patterns
                    .iter()
                    .map(|(pattern, subschema)| (pattern.clone(), subschema.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn key_matches_property_names(
        property_names_pattern: Option<&str>,
        key: &str,
    ) -> Result<bool, GlrMaskError> {
        match property_names_pattern {
            Some(pattern) => Self::schema_pattern_matches_key(pattern, key),
            None => Ok(true),
        }
    }

    fn matching_pattern_schemas_for_key(
        pattern_properties: &[(String, Value)],
        key: &str,
    ) -> Result<Vec<Value>, GlrMaskError> {
        let mut matching = Vec::new();
        for (pattern, subschema) in pattern_properties {
            if Self::schema_pattern_matches_key(pattern, key)? {
                matching.push(subschema.clone());
            }
        }
        Ok(matching)
    }

    fn build_repeated_object_pairs(&self, pair: GrammarExpr) -> GrammarExpr {
        self.build_repeated_object_pairs_with_bounds(pair, 0, None)
    }

    fn build_repeated_object_pairs_with_bounds(
        &self,
        pair: GrammarExpr,
        min_pairs: usize,
        max_pairs: Option<usize>,
    ) -> GrammarExpr {
        if max_pairs == Some(0) {
            return sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]);
        }

        let body = GrammarExpr::SeparatedSequence {
            items: vec![(repeat_expr(pair, min_pairs, max_pairs), true)],
            separator: Box::new(self.json_item_separator_expr()),
            allow_empty: min_pairs == 0,
        };

        sequence_or_single(vec![
            literal_expr(b"{"),
            body,
            literal_expr(b"}"),
        ])
    }

    fn build_repeated_dynamic_object_pairs(
        &mut self,
        value_expr: GrammarExpr,
        min_pairs: usize,
        max_pairs: Option<usize>,
    ) -> GrammarExpr {
        let pair = sequence_or_single(vec![self.json_key_colon_ref(), value_expr]);
        choice_or_single(vec![
            self.build_repeated_object_pairs_with_bounds(pair, min_pairs, max_pairs),
        ])
    }

    fn build_object_pair_tail(
        &mut self,
        base_name: &str,
        suffix: &str,
        pair_exprs: Vec<GrammarExpr>,
        next_nc: GrammarExpr,
        next_c: GrammarExpr,
    ) -> (String, String) {
        let nc_name = format!("{base_name}_{suffix}_nc");
        let c_name = format!("{base_name}_{suffix}_c");

        if pair_exprs.is_empty() {
            self.insert_rule(nc_name.clone(), next_nc);
            self.insert_rule(c_name.clone(), next_c);
            return (nc_name, c_name);
        }

        let pair_expr = choice_or_single(pair_exprs);

        // Use left-recursive repetition instead of right-recursive tail.
        // Old (right-recursive):
        //   c_name → ", " pair c_name | next_c
        // New (left-recursive):
        //   items → ε | items ", " pair
        //   c_name → items next_c
        //
        // This reduces O(N) cascading reduces at closing brace to O(1).
        let items_name = format!("{base_name}_{suffix}_lr");
        self.insert_rule(
            items_name.clone(),
            choice_or_single(vec![
                empty_expr(),
                sequence_or_single(vec![
                    GrammarExpr::Ref(items_name.clone()),
                    self.json_item_separator_expr(),
                    pair_expr.clone(),
                ]),
            ]),
        );

        // c_name → items next_c (or just items if next_c is empty)
        let next_c_is_empty = matches!(&next_c, GrammarExpr::Sequence(s) if s.is_empty());
        if next_c_is_empty {
            self.insert_rule(
                c_name.clone(),
                GrammarExpr::Ref(items_name),
            );
        } else {
            self.insert_rule(
                c_name.clone(),
                sequence_or_single(vec![
                    GrammarExpr::Ref(items_name),
                    next_c,
                ]),
            );
        }

        // nc_name → pair c_name | next_nc (unchanged structure)
        self.insert_rule(
            nc_name.clone(),
            choice_or_single(vec![
                sequence_or_single(vec![pair_expr, GrammarExpr::Ref(c_name.clone())]),
                next_nc,
            ]),
        );

        (nc_name, c_name)
    }

    fn extract_terminal_rule(&mut self, expr: GrammarExpr, prefix: &str) -> GrammarExpr {
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

    /// Extract a terminal rule even if it's trivial (like a literal or raw regex).
    /// Used for exclusions to ensure they resolve to a named terminal reference,
    /// avoiding inline `- /regex/` or `- "literal"` subtraction syntax.
    fn force_extract_terminal_rule(&mut self, expr: GrammarExpr, prefix: &str) -> GrammarExpr {
        if matches!(expr, GrammarExpr::Ref(_)) {
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

    fn schema_all_of(schemas: Vec<Value>) -> Value {
        match schemas.len() {
            0 => serde_json::json!({}),
            1 => schemas.into_iter().next().unwrap(),
            _ => {
                let mut out = Map::new();
                out.insert("allOf".into(), Value::Array(schemas));
                Value::Object(out)
            }
        }
    }

    fn collect_nonempty_index_subsets(
        n: usize,
        start: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        for i in start..n {
            current.push(i);
            out.push(current.clone());
            Self::collect_nonempty_index_subsets(n, i + 1, current, out);
            current.pop();
        }
    }

    fn schema_pattern_matches_key(pattern: &str, key: &str) -> Result<bool, GlrMaskError> {
        let regex = build_regex(&[parse_regex(&json_search_pattern(pattern), true)]);
        let mut state = 0u32;
        for &byte in key.as_bytes() {
            let Some(next) = regex.step(state, byte) else {
                return Ok(false);
            };
            state = next;
        }
        Ok(regex.dfa.finalizers(state).contains(0))
    }

    fn scoped_key_colon_expr(property_names: Option<&Value>) -> Result<LexerExpr, GlrMaskError> {
        let pattern = if let Some(property_names) = property_names {
            key_colon_body_regex(
                &json_search_pattern(Self::property_name_pattern(property_names)?),
            )
        } else {
            key_colon_body_regex(JSON_STRING_BODY_ONLY_REGEX)
        };
        Ok(parse_regex(&pattern, true))
    }

    fn literal_key_colon_union_expr(keys: &BTreeSet<String>) -> Option<LexerExpr> {
        if keys.is_empty() {
            return None;
        }
        let exprs = keys
            .iter()
            .map(|key| LexerExpr::U8Seq(key_colon_literal_body_bytes(key)))
            .collect::<Vec<_>>();
        Some(if exprs.len() == 1 {
            exprs.into_iter().next().unwrap()
        } else {
            LexerExpr::Choice(exprs)
        })
    }

    fn pattern_key_colon_body_expr(&mut self, pattern: &str, prefix: &str) -> GrammarExpr {
        let branches = split_top_level_regex_branches(pattern);
        let mut options = Vec::with_capacity(branches.len());

        for branch in branches {
            let (anchored_start, anchored_end, core) = strip_branch_outer_anchors(branch);
            let inner = jsonify_regex_dot(core);

            let mut parts = Vec::new();
            if !anchored_start {
                parts.push(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
            }

            if core.is_empty() {
                parts.push(empty_expr());
            } else {
                parts.push(regex_expr(&inner));
            }

            if !anchored_end {
                parts.push(GrammarExpr::Repeat(Box::new(self.json_string_char_ref())));
            }

            options.push(sequence_or_single(parts));
        }

        self.extract_terminal_rule(choice_or_single(options), prefix)
    }

    fn build_pattern_key_colon_expr(&mut self, pattern: &str, prefix: &str) -> GrammarExpr {
        let body = self.pattern_key_colon_body_expr(pattern, &format!("{prefix}_BODY"));
        let (terminal_body, wrap) = wrap_key_colon_expr_parts(body);
        let term = self.extract_terminal_rule(terminal_body, prefix);
        wrap(term)
    }

    fn build_json_wrapped_pattern(&mut self, pattern: &str, prefix: &str) -> GrammarExpr {
        let inner = json_search_pattern(pattern);
        let (terminal_body, wrap) = wrap_string_value_expr_parts(parsed_regex_expr(&inner, true));
        let term = self.extract_terminal_rule(terminal_body, prefix);
        wrap(term)
    }

    fn build_json_wrapped_fullmatch_pattern(
        &mut self,
        pattern: &str,
        prefix: &str,
    ) -> GrammarExpr {
        let inner = jsonify_regex_dot(pattern);
        let (terminal_body, wrap) = wrap_string_value_expr_parts(parsed_regex_expr(&inner, true));
        let term = self.extract_terminal_rule(terminal_body, prefix);
        wrap(term)
    }

    fn build_state_machine_expr<State, IsAccepting, Transitions>(
        start: State,
        mut is_accepting: IsAccepting,
        mut transitions_for: Transitions,
    ) -> LexerExpr
    where
        State: Copy + Eq + std::hash::Hash,
        IsAccepting: FnMut(State) -> bool,
        Transitions: FnMut(State) -> Vec<(u8, State)>,
    {
        let mut state_ids = HashMap::<State, usize>::new();
        let mut worklist = VecDeque::<State>::new();
        let mut transitions = Vec::<Vec<(u8, usize)>>::new();
        let mut accepting = Vec::<bool>::new();

        state_ids.insert(start, 0);
        worklist.push_back(start);
        transitions.push(Vec::new());
        accepting.push(false);

        while let Some(product_state) = worklist.pop_front() {
            let result_state_id = state_ids[&product_state];
            accepting[result_state_id] = is_accepting(product_state);

            let mut entries = Vec::new();
            for (byte, next_state) in transitions_for(product_state) {
                let next_result_state_id = if let Some(&existing) = state_ids.get(&next_state) {
                    existing
                } else {
                    let new_state_id = state_ids.len();
                    state_ids.insert(next_state, new_state_id);
                    worklist.push_back(next_state);
                    transitions.push(Vec::new());
                    accepting.push(false);
                    new_state_id
                };
                entries.push((byte, next_result_state_id));
            }
            transitions[result_state_id] = entries;
        }

        let n = transitions.len();
        let final_idx = n;
        let size = n + 1;
        let mut r: Vec<Vec<Option<LexerExpr>>> = vec![vec![None; size]; size];

        for (state_id, entries) in transitions.into_iter().enumerate() {
            let mut by_target: HashMap<usize, crate::ds::u8set::U8Set> = HashMap::new();
            for (byte, target) in entries {
                by_target
                    .entry(target)
                    .or_insert_with(crate::ds::u8set::U8Set::empty)
                    .insert(byte);
            }
            for (target, set) in by_target {
                Self::add_option_expr(&mut r[state_id][target], LexerExpr::U8Class(set));
            }
            if accepting[state_id] {
                Self::add_option_expr(&mut r[state_id][final_idx], LexerExpr::Epsilon);
            }
        }

        for k in (1..n).rev() {
            let kk_star = r[k][k].take().map(|loop_expr| LexerExpr::Repeat {
                expr: Box::new(loop_expr),
                min: 0,
                max: None,
            });

            let sources: Vec<(usize, LexerExpr)> = (0..size)
                .filter(|&i| i != k)
                .filter_map(|i| r[i][k].take().map(|e| (i, e)))
                .collect();

            let targets: Vec<(usize, Option<LexerExpr>)> = (0..size)
                .filter(|&j| j != k)
                .map(|j| (j, r[k][j].clone()))
                .collect();

            for (i, ik_expr) in sources {
                for (j, kj_opt) in &targets {
                    let Some(kj_expr) = kj_opt.clone() else {
                        continue;
                    };
                    let bridge = match &kk_star {
                        Some(star) => LexerExpr::make_seq(vec![ik_expr.clone(), star.clone(), kj_expr]),
                        None => LexerExpr::make_seq(vec![ik_expr.clone(), kj_expr]),
                    };
                    Self::add_option_expr(&mut r[i][*j], bridge);
                }
            }
        }

        let base = r[0][final_idx]
            .take()
            .unwrap_or_else(|| LexerExpr::Choice(vec![]));
        if let Some(loop_expr) = r[0][0].take() {
            let star = LexerExpr::Repeat {
                expr: Box::new(loop_expr),
                min: 0,
                max: None,
            };
            LexerExpr::make_seq(vec![star, base])
        } else {
            base
        }
    }

    fn build_lexer_expr(&mut self, expr: &LexerExpr, prefix: &str) -> GrammarExpr {
        self.extract_terminal_rule(expr_to_grammar_expr(expr), prefix)
    }

    fn add_option_expr(slot: &mut Option<LexerExpr>, new_expr: LexerExpr) {
        match slot {
            None => *slot = Some(new_expr),
            Some(existing) => {
                *existing = LexerExpr::make_choice(vec![existing.clone(), new_expr]);
            }
        }
    }

    /// Build a grammar expression for a key-colon DFA, wrapping it with
    /// the split-off literal parts (opening quote, close quote, colon-space)
    /// just like `wrap_key_colon_regex` does for regex-based keys.
    fn build_key_colon_expr(&mut self, expr: &LexerExpr, prefix: &str) -> GrammarExpr {
        self.build_excluding_key_colon_expr_internal(expr.clone(), vec![], prefix)
    }

    /// Recursively convert a `LexerExpr` to a `GrammarExpr`, extracting
    /// each operand of `Intersect` and `Exclude` as a separate named internal
    /// terminal.  This keeps the composite terminal rule readable: instead of
    /// one giant inline blob the body becomes e.g.
    /// `exclude(intersect(KEY_BODY_0, PATTERN_B_0), "a\"")`.
    fn extract_lexer_expr_decomposed(&mut self, expr: LexerExpr, prefix: &str) -> GrammarExpr {
        match expr {
            LexerExpr::Intersect { expr, intersect } => {
                let left = self.extract_lexer_expr_decomposed(*expr, prefix);
                let right = self.extract_lexer_expr_decomposed(*intersect, prefix);
                GrammarExpr::Intersect {
                    expr: Box::new(left),
                    intersect: Box::new(right),
                }
            }
            LexerExpr::Exclude { expr, exclude } => {
                let left = self.extract_lexer_expr_decomposed(*expr, prefix);
                let right = self.extract_lexer_expr_decomposed(*exclude, prefix);
                GrammarExpr::Exclude {
                    expr: Box::new(left),
                    exclude: Box::new(right),
                }
            }
            other => self.extract_terminal_rule(expr_to_grammar_expr(&other), prefix),
        }
    }

    fn build_excluding_key_colon_expr_internal(
        &mut self,
        base_expr: LexerExpr,
        excluded_exprs: Vec<LexerExpr>,
        prefix: &str,
    ) -> GrammarExpr {
        let excluded_grammar_exprs: Vec<GrammarExpr> = excluded_exprs
            .into_iter()
            .map(|e| self.force_extract_terminal_rule(expr_to_grammar_expr(&e), prefix))
            .collect();
        let body_terminal = self.build_excluding_key_body_expr_internal(
            base_expr,
            excluded_grammar_exprs,
            prefix,
        );
        wrap_key_colon_terminal(body_terminal)
    }

    fn build_excluding_key_body_expr_internal(
        &mut self,
        base_expr: LexerExpr,
        excluded_exprs: Vec<GrammarExpr>,
        prefix: &str,
    ) -> GrammarExpr {
        let base_body = self.extract_lexer_expr_decomposed(base_expr, prefix);
        let body = if excluded_exprs.is_empty() {
            base_body
        } else {
            let excluded_refs: Vec<GrammarExpr> = excluded_exprs
                .into_iter()
                .map(|e| self.force_extract_terminal_rule(e, prefix))
                .collect();
            GrammarExpr::Exclude {
                expr: Box::new(base_body),
                exclude: Box::new(choice_or_single(excluded_refs)),
            }
        };
        self.extract_terminal_rule(body, prefix)
    }

    fn build_excluding_key_colon_expr(
        &mut self,
        base_key_colon_body_expr: GrammarExpr,
        excluded_key_colon_body_exprs: Vec<GrammarExpr>,
        prefix: &str,
    ) -> GrammarExpr {
        let excluded_refs: Vec<GrammarExpr> = excluded_key_colon_body_exprs
            .into_iter()
            .map(|e| self.force_extract_terminal_rule(e, prefix))
            .collect();
        let expr = if excluded_refs.is_empty() {
            base_key_colon_body_expr
        } else {
            GrammarExpr::Exclude {
                expr: Box::new(base_key_colon_body_expr),
                exclude: Box::new(choice_or_single(excluded_refs)),
            }
        };
        wrap_key_colon_terminal(self.extract_terminal_rule(expr, prefix))
    }

    fn shared_additional_key_colon_expr(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        if let Some(expr) = &self.shared_ap_key_colon_expr {
            return Ok(expr.clone());
        }

        let expr = if self.shared_ap_literal_keys.is_empty() {
            self.json_key_colon_ref()
        } else {
            let key_expr = Self::scoped_key_colon_expr(None)?;
            let shared_excluded_expr = Self::literal_key_colon_union_expr(&self.shared_ap_literal_keys);
            let mut excluded_exprs: Vec<LexerExpr> = Vec::new();
            if let Some(expr) = shared_excluded_expr {
                excluded_exprs.push(expr);
            }
            self.build_excluding_key_colon_expr_internal(
                key_expr,
                excluded_exprs,
                "AP_SHARED_KEY_COLON",
            )
        };

        self.shared_ap_key_colon_expr = Some(expr.clone());
        Ok(expr)
    }

    fn shared_additional_key_body_expr(&mut self) -> GrammarExpr {
        if let Some(expr) = &self.shared_ap_key_body_expr {
            return expr.clone();
        }

        let excluded_keys: Vec<String> = self.shared_ap_literal_keys.iter().cloned().collect();
        let excluded = excluded_keys
            .iter()
            .map(|key| self.force_extract_terminal_rule(literal_expr(&key_colon_literal_body_bytes(key)), "AP_SHARED_LITERAL_KEY"))
            .collect::<Vec<_>>();

        let body = if excluded.is_empty() {
            GrammarExpr::Ref(JSON_STRING_BODY_RULE.into())
        } else {
            GrammarExpr::Exclude {
                expr: Box::new(GrammarExpr::Ref(JSON_STRING_BODY_RULE.into())),
                exclude: Box::new(choice_or_single(excluded)),
            }
        };

        let expr = self.insert_named_terminal_rule("AP_SHARED_KEY", body);
        self.shared_ap_key_body_expr = Some(expr.clone());
        expr
    }

    fn build_shared_additional_key_body_choice_expr(
        &mut self,
        excluded_literal_keys: &BTreeSet<String>,
        prefix: &str,
    ) -> GrammarExpr {
        let allowed_back_keys: Vec<String> = self
            .shared_ap_literal_keys
            .iter()
            .filter(|key| !excluded_literal_keys.contains(*key))
            .cloned()
            .collect();

        if allowed_back_keys.is_empty() {
            return self.shared_additional_key_body_expr();
        }

        if let Some(rule_name) = self.shared_ap_key_body_rule_cache.get(&allowed_back_keys) {
            return GrammarExpr::Ref(rule_name.clone());
        }

        let mut options = Vec::with_capacity(1 + allowed_back_keys.len());
        options.push(self.shared_additional_key_body_expr());
        options.extend(
            allowed_back_keys
                .iter()
                .map(|key| literal_expr(&key_colon_literal_body_bytes(key))),
        );

        let rule_name = self.fresh_rule_name(prefix);
        self.insert_rule(rule_name.clone(), choice_or_single(options));
        self.shared_ap_key_body_rule_cache
            .insert(allowed_back_keys, rule_name.clone());
        GrammarExpr::Ref(rule_name)
    }

    fn build_shared_additional_key_choice_expr(
        &mut self,
        excluded_literal_keys: &BTreeSet<String>,
        prefix: &str,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let allowed_back_keys: Vec<String> = self
            .shared_ap_literal_keys
            .iter()
            .filter(|key| !excluded_literal_keys.contains(*key))
            .cloned()
            .collect();

        if allowed_back_keys.is_empty() {
            return self.shared_additional_key_colon_expr();
        }

        if let Some(rule_name) = self.shared_ap_key_rule_cache.get(&allowed_back_keys) {
            return Ok(GrammarExpr::Ref(rule_name.clone()));
        }

        let mut options = Vec::with_capacity(1 + allowed_back_keys.len());
        options.push(self.shared_additional_key_colon_expr()?);
        options.extend(allowed_back_keys.iter().map(|key| self.json_key_colon_literal(key)));

        let rule_name = self.fresh_rule_name(prefix);
        self.insert_rule(rule_name.clone(), choice_or_single(options));
        self.shared_ap_key_rule_cache.insert(allowed_back_keys, rule_name.clone());
        Ok(GrammarExpr::Ref(rule_name))
    }

    fn use_shared_additional_key_exclusions(
        &self,
        excluded_literal_keys: &BTreeSet<String>,
    ) -> bool {
        if !shared_ap_key_exclusions_enabled() || ap_key_any_string() {
            return false;
        }

        let allowed_back = self
            .shared_ap_literal_keys
            .len()
            .saturating_sub(excluded_literal_keys.len());
        allowed_back <= SHARED_AP_MAX_ALLOW_BACK_KEYS
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
            let fixed_count = required_keys.len() as u64;
            if min_properties
                .map(|min| fixed_count < min && properties.is_some() && all_keys_required && no_additional)
                .unwrap_or(false)
                || max_properties.map(|max| fixed_count > max).unwrap_or(false)
            {
                return Err(GlrMaskError::GrammarParse(
                    "min/maxProperties constraints are unsatisfiable for fixed required properties"
                        .into(),
                ));
            }
        }

        let additional_schema = self.normalized_additional_properties_schema(additional_properties);

        let only_dynamic_properties = properties.map(|map| map.is_empty()).unwrap_or(true)
            && required_list.is_empty()
            && pattern_properties.is_none()
            && property_names.is_none();

        if only_dynamic_properties {
            let min_pairs = min_properties.map(|value| value as usize).unwrap_or(0);
            let max_pairs = max_properties.map(|value| value as usize);
            let Some(schema) = additional_schema.clone() else {
                if min_pairs > 0 {
                    return Err(unsat_schema_error());
                }
                return Ok(sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]));
            };

            let value_expr = self.convert_schema(&schema)?;
            return Ok(self.build_repeated_dynamic_object_pairs(value_expr, min_pairs, max_pairs));
        }

        if let Some(properties) = properties {
            if min_properties.is_some() || max_properties.is_some() {
                if let Some(expr) = self.build_min_max_properties_special_case(
                    properties,
                    &required_list,
                    &required_keys,
                    min_properties,
                    max_properties,
                    pattern_properties,
                    additional_properties,
                    property_names,
                )? {
                    return Ok(expr);
                }
                if !all_keys_required || !no_additional {
                    return Err(GlrMaskError::GrammarParse(
                        "min/maxProperties only supported when all keys listed in properties are required"
                            .into(),
                    ));
                }
            }

            let pattern_property_entries = Self::pattern_property_entries(pattern_properties);

            return self.build_ordered_properties_object_expr(
                properties,
                &required_list,
                &required_keys,
                &pattern_property_entries,
                additional_schema,
                property_names,
            );
        }

        if !required_list.is_empty() && pattern_properties.is_none() && property_names.is_none() {
            return self.build_required_any_order_object_expr(&required_list, additional_schema);
        }

        let pattern_property_entries = Self::pattern_property_entries(pattern_properties);

        let empty_props = Map::new();

        if pattern_properties.is_none() {
            if let Some(Value::Object(map)) = additional_properties {
                return self.build_ordered_properties_object_expr(
                    &empty_props,
                    &[],
                    &BTreeSet::new(),
                    &[],
                    Some(Value::Object(map.clone())),
                    property_names,
                );
            }

            if let Some(property_names) = property_names {
                return self.build_pattern_named_object_expr(property_names, self.json_value_ref());
            }

            return Ok(self.json_object_ref());
        }

        if property_names.is_none()
            && pattern_properties.map(|patterns| patterns.len() == 1).unwrap_or(false)
        {
            let (pattern, value_schema) = pattern_properties
                .and_then(|patterns| patterns.iter().next())
                .ok_or_else(|| GlrMaskError::GrammarParse("invalid patternProperties".into()))?;
            let match_all_pattern = pattern == "^.*$" || pattern == ".*";
            let matched_property_names = serde_json::json!({"pattern": pattern});
            let value_expr = self.convert_schema(value_schema)?;
            if matches!(additional_properties, Some(Value::Bool(false))) || match_all_pattern {
                return self.build_pattern_named_object_expr(&matched_property_names, value_expr);
            }

            let additional_value_expr = match additional_properties {
                Some(Value::Object(map)) => self.convert_schema(&Value::Object(map.clone()))?,
                Some(Value::Bool(true)) | None => self.json_value_ref(),
                _ => return Ok(self.json_object_ref()),
            };
            let unmatched_key_colon_body = if let Some(property_names) = property_names {
                regex_expr(key_colon_body_regex(&json_search_pattern(
                    Self::property_name_pattern(property_names)?,
                )))
            } else {
                Self::json_key_colon_full_expr()
            };
            return self.build_mixed_pattern_named_object_expr(
                &matched_property_names,
                value_expr,
                unmatched_key_colon_body,
                additional_value_expr,
            );
        }

        self.build_ordered_properties_object_expr(
            &empty_props,
            &required_list,
            &required_keys,
            &pattern_property_entries,
            additional_schema,
            property_names,
        )
    }

    fn build_required_any_order_object_expr(
        &mut self,
        required_list: &[String],
        additional_properties_schema: Option<Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_req_any_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_nc_0")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;

        let full_mask: Vec<usize> = (0..required_list.len()).collect();
        let mut masks = Vec::<Vec<usize>>::new();
        let mut mask_indices = BTreeMap::<Vec<usize>, usize>::new();
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

        let extra_tail = if let Some(schema) = additional_properties_schema {
            let extra_value_expr = self.convert_schema(&schema)?;
            let ck_prefix = format!("{}_CK", base_name.to_uppercase());
            let extra_key_expr = self.build_excluding_key_colon_expr(
                Self::json_key_colon_full_expr(),
                Vec::new(),
                &format!("{ck_prefix}_KEY_COLON"),
            );
            Some(self.build_object_pair_tail(
                &base_name,
                "ap",
                vec![sequence_or_single(vec![extra_key_expr, extra_value_expr])],
                empty_expr(),
                empty_expr(),
            ))
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

            if mask.is_empty() {
                if let Some((extra_nc, extra_c)) = &extra_tail {
                    nc_alts.push(GrammarExpr::Ref(extra_nc.clone()));
                    c_alts.push(GrammarExpr::Ref(extra_c.clone()));
                } else {
                    nc_alts.push(empty_expr());
                    c_alts.push(empty_expr());
                }
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

    fn build_literal_properties_any_order_object_expr(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        additional_properties_schema: Option<Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let mut literal_entries = Vec::<(String, GrammarExpr, bool)>::new();

        for (key, subschema) in properties {
            let value_expr = match self.convert_schema(subschema) {
                Ok(expr) => expr,
                Err(err) if is_unsat_schema_error(&err) => {
                    if required_keys.contains(key) {
                        return Err(unsat_schema_error());
                    }
                    continue;
                }
                Err(err) => return Err(err),
            };
            literal_entries.push((key.clone(), value_expr, required_keys.contains(key)));
        }

        for key in required_list {
            if properties.contains_key(key) {
                continue;
            }
            let Some(schema) = additional_properties_schema.as_ref() else {
                return Err(unsat_schema_error());
            };
            let value_expr = self.convert_schema(schema)?;
            literal_entries.push((key.clone(), value_expr, true));
        }

        if literal_entries.is_empty() && additional_properties_schema.is_none() {
            return Ok(sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]));
        }

        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_lit_any_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_nc_0")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;

        let full_mask: Vec<usize> = (0..literal_entries.len()).collect();
        let mut masks = Vec::<Vec<usize>>::new();
        let mut mask_indices = BTreeMap::<Vec<usize>, usize>::new();
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

        let extra_pair = if let Some(schema) = additional_properties_schema {
            let extra_value_expr = self.convert_schema(&schema)?;
            let mut excluded_literal_keys = BTreeSet::<String>::new();
            excluded_literal_keys.extend(properties.keys().cloned());
            excluded_literal_keys.extend(required_list.iter().cloned());
            if ap_key_any_string() {
                // Skip key exclusion — AP keys accept any JSON string.
                // Each object still gets its own uniquely-named terminal.
                let full_key_expr = Self::scoped_key_colon_expr(None)?;
                Some(sequence_or_single(vec![
                    self.build_key_colon_expr(
                        &full_key_expr,
                        &format!("{}_KEY_COLON", base_name.to_uppercase()),
                    ),
                    extra_value_expr,
                ]))
            } else if self.use_shared_additional_key_exclusions(&excluded_literal_keys) {
                Some(sequence_or_single(vec![
                    self.build_shared_additional_key_choice_expr(
                        &excluded_literal_keys,
                        &format!("{base_name}_ap_key"),
                    )?,
                    extra_value_expr,
                ]))
            } else {
                let key_expr = Self::scoped_key_colon_expr(None)?;
                let mut excluded_exprs: Vec<LexerExpr> = Vec::new();
                if let Some(expr) = Self::literal_key_colon_union_expr(&excluded_literal_keys) {
                    excluded_exprs.push(expr);
                }
                Some(sequence_or_single(vec![
                    self.build_excluding_key_colon_expr_internal(
                        key_expr,
                        excluded_exprs,
                        &format!("{}_KEY_COLON", base_name.to_uppercase()),
                    ),
                    extra_value_expr,
                ]))
            }
        } else {
            None
        };

        for mask in &masks {
            let mask_index = *mask_indices.get(mask).unwrap();
            let nc_name = format!("{base_name}_nc_{mask_index}");
            let c_name = format!("{base_name}_c_{mask_index}");
            let mut nc_alts = Vec::new();
            let mut c_alts = Vec::new();

            if let Some(extra_pair) = &extra_pair {
                nc_alts.push(sequence_or_single(vec![
                    extra_pair.clone(),
                    GrammarExpr::Ref(c_name.clone()),
                ]));
                c_alts.push(sequence_or_single(vec![
                    self.json_item_separator_expr(),
                    extra_pair.clone(),
                    GrammarExpr::Ref(c_name.clone()),
                ]));
            }

            for &item in mask {
                let (key, value_expr, _) = &literal_entries[item];
                let pair_expr = sequence_or_single(vec![
                    self.json_key_colon_literal(key),
                    value_expr.clone(),
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

            let missing_required = mask.iter().any(|&item| literal_entries[item].2);
            if !missing_required {
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

    fn supports_literal_properties_any_order_fast_path(expr: &GrammarExpr) -> bool {
        matches!(
            expr,
            GrammarExpr::Ref(rule_name)
                if matches!(
                    rule_name.as_str(),
                    JSON_BOOL_RULE | JSON_INTEGER_RULE | JSON_NUMBER_RULE | JSON_NULL_RULE
                )
        ) || matches!(expr, GrammarExpr::Literal(_))
    }

    fn build_array_item_sequence(
        &mut self,
        items: &[(GrammarExpr, bool)],
        needs_separator: bool,
        cache: &mut BTreeMap<(usize, bool), GrammarExpr>,
        index: usize,
    ) -> GrammarExpr {
        if let Some(expr) = cache.get(&(index, needs_separator)) {
            return expr.clone();
        }
        if index >= items.len() {
            let result = empty_expr();
            cache.insert((index, needs_separator), result.clone());
            return result;
        }

        let (item_expr, required) = &items[index];
        let remaining_with_separator = self.build_array_item_sequence(items, true, cache, index + 1);
        let item_and_rest = if needs_separator {
            sequence_or_single(vec![
                self.json_item_separator_expr(),
                item_expr.clone(),
                remaining_with_separator,
            ])
        } else {
            sequence_or_single(vec![item_expr.clone(), remaining_with_separator])
        };
        let result = if *required {
            item_and_rest
        } else {
            choice_or_single(vec![item_and_rest, empty_expr()])
        };
        cache.insert((index, needs_separator), result.clone());
        result
    }

    fn normalize_array_keywords(
        &self,
        schema: &Map<String, Value>,
    ) -> (Vec<Value>, Option<Value>, usize, Option<usize>) {
        let prefix_items = schema.get("prefixItems");
        let items = schema.get("items");
        let additional_items = schema.get("additionalItems");

        let (effective_prefix_items, effective_items) = if self.current_draft()
            <= JsonSchemaDraft::Draft201909
            || additional_items.is_some()
            || matches!(items, Some(Value::Array(_)))
        {
            match items {
                Some(Value::Array(values)) => (values.clone(), additional_items.cloned()),
                _ => (Vec::new(), items.cloned()),
            }
        } else {
            (
                prefix_items
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                items.cloned(),
            )
        };

        let min_items = schema
            .get("minItems")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(0);
        let max_items = schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        (effective_prefix_items, effective_items, min_items, max_items)
    }

    fn build_min_max_properties_special_case(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        min_properties: Option<u64>,
        max_properties: Option<u64>,
        pattern_properties: Option<&Map<String, Value>>,
        additional_properties: Option<&Value>,
        property_names: Option<&Value>,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if properties.is_empty() {
            return Ok(None);
        }
        if pattern_properties.map(|patterns| !patterns.is_empty()).unwrap_or(false) {
            return Ok(None);
        }

        let closed_object = matches!(additional_properties, Some(Value::Bool(false)))
            || additional_properties
                .map(|schema| self.is_certainly_unsatisfiable(schema))
                .unwrap_or(false);
        if !closed_object {
            return Ok(None);
        }

        let fixed_required_count = required_keys.len() as u64;
        let residual_min = min_properties
            .map(|min| min.saturating_sub(fixed_required_count))
            .unwrap_or(0);
        let residual_max = max_properties.map(|max| max.saturating_sub(fixed_required_count));
        let optional_keys = properties
            .keys()
            .filter(|key| !required_keys.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        if optional_keys.is_empty() || (residual_min == 0 && residual_max.is_none()) {
            return Ok(None);
        }

        let supports_exactly_or_at_most_one = residual_min <= 1 && residual_max == Some(1);
        let supports_at_least_one = residual_min >= 1 && residual_max.is_none();
        if !(supports_exactly_or_at_most_one || supports_at_least_one) {
            return Ok(None);
        }

        let empty_patterns: Vec<(String, Value)> = Vec::new();
        let mut options = Vec::new();
        if supports_exactly_or_at_most_one {
            for optional_key in &optional_keys {
                let allowed_properties = properties
                    .iter()
                    .filter(|(key, _)| required_keys.contains(*key) || *key == optional_key)
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<Map<_, _>>();
                let option_required = required_list
                    .iter()
                    .cloned()
                    .chain(std::iter::once(optional_key.clone()))
                    .collect::<Vec<_>>();
                let option_required_keys = option_required.iter().cloned().collect::<BTreeSet<_>>();
                options.push(self.build_ordered_properties_object_expr(
                    &allowed_properties,
                    &option_required,
                    &option_required_keys,
                    &empty_patterns,
                    None,
                    property_names,
                )?);
            }
            if residual_min == 0 {
                let required_only = properties
                    .iter()
                    .filter(|(key, _)| required_keys.contains(*key))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<Map<_, _>>();
                options.push(self.build_ordered_properties_object_expr(
                    &required_only,
                    required_list,
                    required_keys,
                    &empty_patterns,
                    None,
                    property_names,
                )?);
            }
            return Ok(Some(choice_or_single(options)));
        }

        Ok(Some(self.build_ordered_properties_object_expr_with_options(
            properties,
            required_list,
            required_keys,
            &empty_patterns,
            None,
            property_names,
            false,
        )?))
    }

    fn build_ordered_properties_object_expr_with_options(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        pattern_properties: &[(String, Value)],
        additional_properties_schema: Option<Value>,
        property_names: Option<&Value>,
        allow_empty_named_props_list: bool,
    ) -> Result<GrammarExpr, GlrMaskError> {
        self.build_ordered_properties_object_expr_impl(
            properties,
            required_list,
            required_keys,
            pattern_properties,
            additional_properties_schema,
            property_names,
            allow_empty_named_props_list,
        )
    }

    fn build_ordered_properties_object_expr(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        pattern_properties: &[(String, Value)],
        additional_properties_schema: Option<Value>,
        property_names: Option<&Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        self.build_ordered_properties_object_expr_impl(
            properties,
            required_list,
            required_keys,
            pattern_properties,
            additional_properties_schema,
            property_names,
            true,
        )
    }

    fn build_ordered_properties_object_expr_impl(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        pattern_properties: &[(String, Value)],
        additional_properties_schema: Option<Value>,
        property_names: Option<&Value>,
        allow_empty_named_props_list: bool,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let property_names_pattern = property_names
            .map(Self::property_name_pattern)
            .transpose()?
            .map(str::to_string);

        // Exclude all literal fixed keys from the pattern-based free-form branch,
        // even if their individual schema becomes unsatisfiable. A key named in
        // `properties` is never a pattern-driven fallback property.
        let mut fixed_literal_keys = BTreeSet::<String>::new();
        fixed_literal_keys.extend(properties.keys().cloned());
        fixed_literal_keys.extend(required_list.iter().cloned());

        let mut ordered: Vec<(String, GrammarExpr, bool)> = Vec::new();

        for (key, subschema) in properties {
            if !Self::key_matches_property_names(property_names_pattern.as_deref(), key)? {
                if required_keys.contains(key) {
                    return Err(unsat_schema_error());
                }
                continue;
            }

            let mut schemas = vec![subschema.clone()];
            schemas.extend(Self::matching_pattern_schemas_for_key(pattern_properties, key)?);

            let effective_schema = Self::schema_all_of(schemas);
            let value_expr = match self.convert_schema(&effective_schema) {
                Ok(expr) => expr,
                Err(err) if is_unsat_schema_error(&err) => {
                    if required_keys.contains(key) {
                        return Err(unsat_schema_error());
                    }
                    continue;
                }
                Err(err) => return Err(err),
            };

            ordered.push((key.clone(), value_expr, required_keys.contains(key)));
        }

        for key in required_list {
            if properties.contains_key(key) {
                continue;
            }

            if !Self::key_matches_property_names(property_names_pattern.as_deref(), key)? {
                return Err(unsat_schema_error());
            }

            let matching_pattern_schemas =
                Self::matching_pattern_schemas_for_key(pattern_properties, key)?;

            let effective_schema = if matching_pattern_schemas.is_empty() {
                match &additional_properties_schema {
                    Some(schema) => schema.clone(),
                    None => return Err(unsat_schema_error()),
                }
            } else {
                Self::schema_all_of(matching_pattern_schemas)
            };

            let value_expr = match self.convert_schema(&effective_schema) {
                Ok(expr) => expr,
                Err(err) if is_unsat_schema_error(&err) => return Err(unsat_schema_error()),
                Err(err) => return Err(err),
            };

            ordered.push((key.clone(), value_expr, true));
        }

        if ordered.is_empty() && additional_properties_schema.is_none() && pattern_properties.is_empty() {
            return Ok(sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]));
        }

        if pattern_properties.is_empty()
            && property_names.is_none()
            && additional_properties_schema.is_some()
            && ordered.iter().all(|(_, _, required)| !*required)
            && ordered
                .iter()
                .all(|(_, value_expr, _)| Self::supports_literal_properties_any_order_fast_path(value_expr))
        {
            return self.build_literal_properties_any_order_object_expr(
                properties,
                required_list,
                required_keys,
                additional_properties_schema,
            );
        }

        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_ord_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_obj")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;

        let upper_base_name = base_name.to_uppercase();
        let np_terminal = Self::literal_key_colon_union_expr(&fixed_literal_keys).map(|expr| {
            self.insert_named_terminal_rule(
                format!("{upper_base_name}_NP"),
                expr_to_grammar_expr(&expr),
            )
        });

        let mut pp_kv_rules = Vec::new();
        let mut pattern_key_terminals = Vec::<GrammarExpr>::new();
        if !pattern_properties.is_empty() {
            for (pattern_idx, (pattern, pattern_schema)) in pattern_properties.iter().enumerate() {
                let body = self.pattern_key_colon_body_expr(pattern, &format!("{}_PP{}_KEY_BODY", upper_base_name, pattern_idx));
                let (terminal_body, wrap) = wrap_key_colon_expr_parts(body);
                let term_ref = self.extract_terminal_rule(terminal_body, &format!("{}_PP{}_KEY", upper_base_name, pattern_idx));
                let key_expr = wrap(term_ref.clone());

                let value_expr = match self.convert_schema(pattern_schema) {
                    Ok(expr) => expr,
                    Err(err) if is_unsat_schema_error(&err) => continue,
                    Err(err) => return Err(err),
                };

                let kv_rule = self.insert_rule(
                    format!("{base_name}_pp_kv_{pattern_idx}"),
                    sequence_or_single(vec![key_expr, value_expr]),
                );
                pp_kv_rules.push(GrammarExpr::Ref(kv_rule));
                pattern_key_terminals.push(term_ref);
            }
        }

        let pattern_terminal = if pattern_key_terminals.is_empty() {
            None
        } else {
            Some(self.insert_named_terminal_rule(
                format!("{upper_base_name}_PP"),
                choice_or_single(pattern_key_terminals),
            ))
        };

        let mut ap_kv_rule = None;
        let has_additional_properties = additional_properties_schema.is_some();
        if let Some(schema) = additional_properties_schema {
            let ap_key_expr = if self.use_shared_additional_key_exclusions(&fixed_literal_keys) {
                let shared_choice = self.build_shared_additional_key_body_choice_expr(
                    &fixed_literal_keys,
                    &format!("{base_name}_ap_key"),
                );

                if let Some(pattern_terminal) = &pattern_terminal {
                    self.insert_named_terminal_rule(
                        format!("{upper_base_name}_AP_SHARED_FILTERED"),
                        GrammarExpr::Exclude {
                            expr: Box::new(shared_choice),
                            exclude: Box::new(pattern_terminal.clone()),
                        },
                    )
                } else {
                    shared_choice
                }
            } else {
                let mut excluded_ap_exprs = Vec::<GrammarExpr>::new();
                if let Some(np_terminal) = &np_terminal {
                    excluded_ap_exprs.push(np_terminal.clone());
                }
                if let Some(pattern_terminal) = &pattern_terminal {
                    excluded_ap_exprs.push(pattern_terminal.clone());
                }

                let ap_body = if excluded_ap_exprs.is_empty() {
                    GrammarExpr::Ref(JSON_STRING_BODY_RULE.into())
                } else {
                    GrammarExpr::Exclude {
                        expr: Box::new(GrammarExpr::Ref(JSON_STRING_BODY_RULE.into())),
                        exclude: Box::new(choice_or_single(excluded_ap_exprs)),
                    }
                };
                self.insert_named_terminal_rule(format!("{upper_base_name}_AP"), ap_body)
            };

            let wrapped_ap_key = wrap_key_colon_terminal(ap_key_expr);
            let additional_value_expr = self.convert_schema(&schema)?;
            let kv_rule = self.insert_shared_rule(
                format!("{base_name}_ap_kv"),
                sequence_or_single(vec![wrapped_ap_key, additional_value_expr]),
            );
            ap_kv_rule = Some(GrammarExpr::Ref(kv_rule));
        }

        let pattern_list_expr = if pp_kv_rules.is_empty() {
            None
        } else {
            let pp_alt = self.insert_shared_rule(
                format!("{base_name}_pp_alt"),
                choice_or_single(pp_kv_rules),
            );
            Some(GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::Repeat(Box::new(GrammarExpr::Ref(pp_alt))), true)],
                separator: Box::new(self.json_item_separator_expr()),
                allow_empty: true,
            })
        };

        let additional_list_expr = ap_kv_rule.clone().map(|pair| {
            GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::Repeat(Box::new(pair)), true)],
                separator: Box::new(self.json_item_separator_expr()),
                allow_empty: true,
            }
        });



        if !ordered.is_empty()
            && ordered.iter().all(|(_, _, required)| *required)
            && pattern_properties.is_empty()
            && !has_additional_properties
        {
            return Ok(self.build_closed_required_ordered_object_expr(&ordered));
        }

        let pattern_list_rule = pattern_list_expr.map(|expr| {
            self.insert_shared_rule(format!("{base_name}_pp_list"), expr)
        });
        let additional_list_rule = additional_list_expr.map(|expr| {
            self.insert_shared_rule(format!("{base_name}_ap_list"), expr)
        });
        let additional_nonempty_list_rule = ap_kv_rule.clone().map(|pair| {
            self.insert_shared_rule(
                format!("{base_name}_ap_list_nonempty"),
                GrammarExpr::SeparatedSequence {
                    items: vec![(GrammarExpr::RepeatOne(Box::new(pair)), true)],
                    separator: Box::new(self.json_item_separator_expr()),
                    allow_empty: true,
                },
            )
        });

        let mut np_item_rules = Vec::<(String, bool)>::new();
        for (idx, (key, value_expr, is_required)) in ordered.iter().enumerate() {
            let np_item_rule = self.insert_shared_rule(
                format!("{base_name}_np_{idx}"),
                self.build_merged_literal_key_value_expr(b"", key, value_expr.clone()),
            );
            np_item_rules.push((np_item_rule, *is_required));
        }
        let named_props_list_rule = if np_item_rules.is_empty() {
            None
        } else {
            Some(self.insert_shared_rule(
                format!("{base_name}_np_list"),
                GrammarExpr::SeparatedSequence {
                    items: np_item_rules
                        .iter()
                        .map(|(rule_name, is_required)| {
                            (GrammarExpr::Ref(rule_name.clone()), *is_required)
                        })
                        .collect(),
                    separator: Box::new(self.json_item_separator_expr()),
                    allow_empty: allow_empty_named_props_list,
                },
            ))
        };

        let mut body_items: Vec<(GrammarExpr, bool)> = Vec::new();
        if let Some(np_rule) = &named_props_list_rule {
            body_items.push((GrammarExpr::Ref(np_rule.clone()), true));
        }
        if let Some(pp_rule) = &pattern_list_rule {
            body_items.push((GrammarExpr::Ref(pp_rule.clone()), true));
        }
        if let Some(ap_rule) = &additional_list_rule {
            body_items.push((GrammarExpr::Ref(ap_rule.clone()), true));
        }

        if body_items.is_empty() {
            return Ok(sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]));
        }

        let np_guaranteed_nonempty = !ordered.is_empty() && ordered.iter().any(|(_, _, required)| *required);

        let body_expr = if pattern_list_rule.is_none()
            && np_guaranteed_nonempty
            && named_props_list_rule.is_some()
            && additional_nonempty_list_rule.is_some()
        {
            // Emit an explicit non-nullable form to avoid transient GLR ambiguity from
            // nullable list composition in SeparatedSequence lowering.
            let np_rule = named_props_list_rule.clone().unwrap();
            let ap_nonempty_rule = additional_nonempty_list_rule.clone().unwrap();
            let options = vec![
                GrammarExpr::Ref(np_rule.clone()),
                sequence_or_single(vec![
                    GrammarExpr::Ref(np_rule),
                    self.json_item_separator_expr(),
                    GrammarExpr::Ref(ap_nonempty_rule.clone()),
                ]),
            ];

            choice_or_single(options)
        } else if body_items.len() == 1 {
            body_items.into_iter().next().unwrap().0
        } else {
            GrammarExpr::SeparatedSequence {
                items: body_items,
                separator: Box::new(self.json_item_separator_expr()),
                allow_empty: true,
            }
        };

        let body_rule = self.insert_shared_rule(format!("{base_name}_body"), body_expr);
        let object_expr = sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(body_rule),
            literal_expr(b"}"),
        ]);

        let object_rule = self.insert_shared_rule(format!("{base_name}_obj"), object_expr);
        Ok(GrammarExpr::Ref(object_rule))
    }

    fn property_name_pattern(property_names: &Value) -> Result<&str, GlrMaskError> {
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
            self.build_pattern_key_colon_expr(pattern, "PP_KEY_COLON"),
            value_expr,
        ]);
        Ok(self.build_repeated_object_pairs(pair))
    }

    fn build_mixed_pattern_named_object_expr(
        &mut self,
        property_names: &Value,
        matched_value_expr: GrammarExpr,
        unmatched_key_colon_body: GrammarExpr,
        unmatched_value_expr: GrammarExpr,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let pattern = Self::property_name_pattern(property_names)?;
        let matched_key_colon_body = self.pattern_key_colon_body_expr(pattern, "PP_KEY_COLON_BODY");
        let (matched_terminal_body, wrap) = wrap_key_colon_expr_parts(matched_key_colon_body.clone());
        let matched_key_colon_full = wrap(self.extract_terminal_rule(matched_terminal_body.clone(), "PP_KEY_COLON"));
        let matched_pair = sequence_or_single(vec![
            matched_key_colon_full,
            matched_value_expr,
        ]);
        let unmatched_pair = sequence_or_single(vec![
            self.build_excluding_key_colon_expr(
                unmatched_key_colon_body,
                vec![matched_terminal_body],
                "PP_KEY_COLON",
            ),
            unmatched_value_expr,
        ]);
        let base_name = self.fresh_rule_name("obj_pat_mix");
        let matched_pair_rule = self.insert_shared_rule(
            format!("{base_name}_pp_kv"),
            matched_pair,
        );
        let unmatched_pair_rule = self.insert_shared_rule(
            format!("{base_name}_ap_kv"),
            unmatched_pair,
        );
        let matched_list_nonempty = self.insert_shared_rule(
            format!("{base_name}_pp_list"),
            GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::RepeatOne(Box::new(GrammarExpr::Ref(matched_pair_rule))), true)],
                separator: Box::new(self.json_item_separator_expr()),
                allow_empty: false,
            },
        );
        let unmatched_list_nonempty = self.insert_shared_rule(
            format!("{base_name}_ap_list"),
            GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::RepeatOne(Box::new(GrammarExpr::Ref(unmatched_pair_rule))), true)],
                separator: Box::new(self.json_item_separator_expr()),
                allow_empty: false,
            },
        );
        let body_rule = self.insert_shared_rule(
            format!("{base_name}_body"),
            choice_or_single(vec![
                empty_expr(),
                GrammarExpr::Ref(matched_list_nonempty.clone()),
                GrammarExpr::Ref(unmatched_list_nonempty.clone()),
                sequence_or_single(vec![
                    GrammarExpr::Ref(matched_list_nonempty),
                    self.json_item_separator_expr(),
                    GrammarExpr::Ref(unmatched_list_nonempty),
                ]),
            ]),
        );
        Ok(sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(body_rule),
            literal_expr(b"}"),
        ]))
    }

    fn build_array_expr(&mut self, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        let (prefix_items, items_schema, min_items, max_items) = self.normalize_array_keywords(schema);

        if max_items.map(|max| min_items > max).unwrap_or(false) {
            return Err(unsat_schema_error());
        }

        if !prefix_items.is_empty() {
            let additional_item_expr = match items_schema {
                Some(Value::Bool(false)) => None,
                Some(Value::Bool(true)) | None => Some(self.json_value_ref()),
                Some(ref value) => match self.convert_schema(value) {
                    Ok(expr) => Some(expr),
                    Err(err) if is_unsat_schema_error(&err) => {
                        if prefix_items.len() >= min_items {
                            None
                        } else {
                            return Err(unsat_schema_error());
                        }
                    }
                    Err(err) => return Err(err),
                },
            };

            let mut array_items = Vec::<(GrammarExpr, bool)>::new();
            let mut effective_max_items = max_items;
            let item_count = effective_max_items.unwrap_or(prefix_items.len().max(min_items));

            for index in 0..item_count {
                let item_expr = if index < prefix_items.len() {
                    match self.convert_schema(&prefix_items[index]) {
                        Ok(expr) => expr,
                        Err(err) if is_unsat_schema_error(&err) => {
                            if index >= min_items {
                                effective_max_items = Some(index);
                                break;
                            }
                            return Err(unsat_schema_error());
                        }
                        Err(err) => return Err(err),
                    }
                } else if let Some(expr) = additional_item_expr.clone() {
                    expr
                } else {
                    break;
                };
                array_items.push((item_expr, index < min_items));
            }

            if effective_max_items.is_none() {
                if let Some(expr) = additional_item_expr {
                    let tail_item = self.extract_rule(expr, "arr_item");
                    let tail = sequence_or_single(vec![
                        tail_item.clone(),
                        repeat_expr(
                            sequence_or_single(vec![self.json_item_separator_expr(), tail_item]),
                            0,
                            None,
                        ),
                    ]);
                    array_items.push((tail, false));
                }
            }

            let mut sequence_cache = BTreeMap::new();
            let body = self.build_array_item_sequence(&array_items, false, &mut sequence_cache, 0);
            return Ok(sequence_or_single(vec![
                literal_expr(b"["),
                body,
                literal_expr(b"]"),
            ]));
        }

        match items_schema {
            Some(Value::Bool(false)) => {
                if min_items > 0 {
                    return Err(unsat_schema_error());
                }
                return Ok(sequence_or_single(vec![literal_expr(b"["), literal_expr(b"]")]));
            }
            Some(Value::Bool(true)) | None => {
                if schema.contains_key("minItems") || schema.contains_key("maxItems") {
                    return Ok(self.build_repeated_array(self.json_value_ref(), min_items, max_items));
                }
                return Ok(self.json_array_ref());
            }
            Some(ref item_schema) if item_schema.is_object() => {
                let item_expr = self.convert_schema(item_schema)?;
                return Ok(self.build_repeated_array(item_expr, min_items, max_items));
            }
            Some(_) => {}
        }

        if schema.contains_key("minItems") || schema.contains_key("maxItems") {
            return Ok(self.build_repeated_array(self.json_value_ref(), min_items, max_items));
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

        let mut array_items = Vec::new();
        match (min_items, max_items) {
            (0, None) => array_items.push((GrammarExpr::Repeat(Box::new(item_expr)), true)),
            (1, None) => array_items.push((GrammarExpr::RepeatOne(Box::new(item_expr)), true)),
            (m, Some(n)) => array_items.push((GrammarExpr::RepeatRange {
                expr: Box::new(item_expr),
                min: m,
                max: n,
            }, true)),
            (m, None) => {
                array_items.push((GrammarExpr::RepeatRange {
                    expr: Box::new(item_expr.clone()),
                    min: m,
                    max: m,
                }, true));
                array_items.push((GrammarExpr::Repeat(Box::new(item_expr)), true));
            }
        }

        let body = GrammarExpr::SeparatedSequence {
            items: array_items,
            separator: Box::new(self.json_item_separator_expr()),
            allow_empty: min_items == 0,
        };
        sequence_or_single(vec![
            literal_expr(b"["),
            body,
            literal_expr(b"]"),
        ])
    }

    fn is_trivial_expr(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(_) | GrammarExpr::Literal(_) | GrammarExpr::RawRegex(_) => true,
            // Empty sequence = epsilon.
            GrammarExpr::Sequence(parts) => parts.is_empty(),
            // A choice between only trivial alternatives needs no named wrapper.
            GrammarExpr::Choice(parts) => !parts.is_empty() && parts.iter().all(Self::is_trivial_expr),
            _ => false,
        }
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

/// Collect terminal names that are referenced from nonterminal rule bodies.
/// These terminals are "grammar-visible" and must have their own TerminalID.
fn collect_grammar_visible_refs(
    rules: &[NamedRule],
    terminal_names: &BTreeSet<String>,
) -> BTreeSet<String> {
    fn walk(expr: &GrammarExpr, terminal_names: &BTreeSet<String>, out: &mut BTreeSet<String>) {
        match expr {
            GrammarExpr::Ref(name) => {
                if terminal_names.contains(name) {
                    out.insert(name.clone());
                }
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                for p in parts {
                    walk(p, terminal_names, out);
                }
            }
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner) => walk(inner, terminal_names, out),
            GrammarExpr::RepeatRange { expr, .. } => walk(expr, terminal_names, out),
            GrammarExpr::Exclude { expr, exclude } => {
                walk(expr, terminal_names, out);
                walk(exclude, terminal_names, out);
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item_expr, _) in items {
                    walk(item_expr, terminal_names, out);
                }
                walk(separator, terminal_names, out);
            }
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::AnyByte
            | GrammarExpr::Intersect { .. } => {}
        }
    }
    let mut visible = BTreeSet::new();
    for rule in rules {
        if !rule.is_terminal {
            walk(&rule.expr, terminal_names, &mut visible);
        }
    }
    visible
}

fn choice_or_single(alts: Vec<GrammarExpr>) -> GrammarExpr {
    let mut seen = BTreeSet::new();
    let mut alts: Vec<GrammarExpr> = alts
        .into_iter()
        .filter(|expr| seen.insert(expr_key(expr)))
        .collect();
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

