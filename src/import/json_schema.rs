use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use serde_json::{Map, Value};
use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr as LexerExpr;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::dfa::DFA as LexerDfa;
use crate::automata::lexer::regex::parse_regex;
use crate::compiler::grammar_def::GrammarDef;
use crate::ds::bitset::BitSet;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, lower, promote_large_literal_alts};

// WARNING: Do NOT break terminals containing repeats of multi-char subexpressions
// into grammar-level repeats of single characters. Doing so creates terminals of
// byte-length 1, which catastrophically bloats the terminal DWA (the parser must
// track every possible single-byte terminal match at every position). Instead,
// keep repeated character patterns fused into chunked multi-char terminals
// (e.g. char{1024}) and use TerminalExpr(Repeat{...}) to trigger the direct
// bounded-repeat DFA construction path, which avoids NFA→DFA blowup.

const JSON_VALUE_RULE: &str = "json_value";
const JSON_OBJECT_RULE: &str = "json_object";
const JSON_ARRAY_RULE: &str = "json_array";
const JSON_KV_RULE: &str = "json_kv";
const JSON_STRING_RULE: &str = "json_string";
const JSON_STRING_BODY_RULE: &str = "JSON_STRING_BODY";
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
const EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS: usize = 12;
const EXACT_CLOSED_OBJECT_UNION_MAX_STATES: usize = 128;

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
        | GrammarExpr::TerminalExpr(_)
        | GrammarExpr::Exclude { .. }
        | GrammarExpr::Repeat(_)
        | GrammarExpr::RepeatOne(_)
        | GrammarExpr::RepeatRange { .. }
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::AnyByte => return None,
    };

    let total_bytes = alts.iter().map(Vec::len).sum::<usize>();
    if alts.len() > max_alts || total_bytes > CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_TOTAL_BYTES {
        return None;
    }

    Some(alts)
}

fn maybe_fuse_finite_literal_expr(expr: GrammarExpr, context: &str) -> GrammarExpr {
    let Some(alts) = finite_literal_alternatives(&expr, CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_ALTS) else {
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

fn strip_single_outer_group(branch: &str) -> Option<&str> {
    let bytes = branch.as_bytes();
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
        Expr::Dfa(_) => (0, None),
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
    std::env::var(name)
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off"))
        .unwrap_or(false)
}

fn split_open_quote() -> bool {
    // Default to true: opening quote is split unless explicitly disabled
    std::env::var("GLRMASK_SPLIT_OPEN_QUOTE")
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

fn split_close_quote() -> bool {
    env_flag("GLRMASK_SPLIT_CLOSE_QUOTE")
}

fn split_colon_space() -> bool {
    // Default to true: colon-space is split unless explicitly disabled
    std::env::var("GLRMASK_SPLIT_COLON_SPACE")
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

fn split_colon_from_space() -> bool {
    env_flag("GLRMASK_SPLIT_COLON_FROM_SPACE")
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

/// Build literal bytes for the body terminal of a JSON string **value**.
///
/// Fuses non-split quotes into the byte sequence.
fn string_value_literal_body_bytes(text: &str) -> Vec<u8> {
    let full = json_string_literal_bytes(text); // b'"text"'
    let body_only = &full[1..full.len() - 1];
    let open = split_open_quote();
    let close = split_close_quote();
    let mut bytes = Vec::new();
    if !open { bytes.push(b'"'); }
    bytes.extend_from_slice(body_only);
    if !close { bytes.push(b'"'); }
    bytes
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

/// Build the colon-space suffix expression for keys: `": "`, `":"` + `" "`, or nothing.
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

/// Wrap a body regex as a JSON string value expression.
/// Shorthand for `wrap_string_value_terminal(regex_expr(string_value_body_regex(inner)))`.
fn wrap_string_value_regex(inner_regex: &str) -> GrammarExpr {
    wrap_string_value_terminal(regex_expr(string_value_body_regex(inner_regex)))
}

/// Wrap a body regex as a JSON key-colon expression.
/// Shorthand for `wrap_key_colon_terminal(regex_expr(key_colon_body_regex(inner)))`.
fn wrap_key_colon_regex(inner_regex: &str) -> GrammarExpr {
    wrap_key_colon_terminal(regex_expr(key_colon_body_regex(inner_regex)))
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
    env_flag("GLRMASK_AP_SHARED_EXCLUSIONS")
}

fn collect_shared_ap_literal_keys(root: &Value) -> BTreeSet<String> {
    let mut collected = BTreeSet::new();
    let mut queue = VecDeque::from([root]);

    while let Some(node) = queue.pop_front() {
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

fn max_string_length_cap() -> Option<usize> {
    std::env::var("GLRMASK_MAX_STRING_LENGTH_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

fn json_wrapped_pattern(pattern: &str) -> GrammarExpr {
    let inner = json_search_pattern(pattern);
    wrap_string_value_regex(&inner)
}

fn json_wrapped_pattern_bounded(pattern: &str, max_tail: usize) -> GrammarExpr {
    let inner = json_search_pattern_bounded(pattern, max_tail);
    wrap_string_value_regex(&inner)
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

fn integer_multiple_dfa(multiple: u64, allow_fractional_zero: bool) -> LexerDfa {
    assert!(multiple > 0);

    SchemaCtx::build_product_lexer_dfa(
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

fn reciprocal_power_of_ten_dfa(scale: usize) -> LexerDfa {
    assert!(scale > 0);

    SchemaCtx::build_product_lexer_dfa(
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

fn compile_regex_union_dfa(regexes: &[String]) -> LexerDfa {
    let exprs = regexes
        .iter()
        .map(|regex| parse_regex(regex, true))
        .collect::<Vec<_>>();
    build_regex(&exprs).dfa
}

fn json_wrapped_fullmatch_pattern(pattern: &str) -> GrammarExpr {
    let inner = jsonify_regex_dot(pattern);
    wrap_string_value_regex(&inner)
}

fn json_wrapped_key_colon_pattern(pattern: &str) -> GrammarExpr {
    let inner = json_search_pattern(pattern);
    wrap_key_colon_regex(&inner)
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
    let year = regex_expr(r#"[0-9]{4}"#);
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
    let day_29 = regex_expr(r#"(?:0[1-9]|1[0-9]|2[0-9])"#);

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

    let t2 = std::time::Instant::now();
    promote_large_literal_alts(&mut named, 10);
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
    let rules: Vec<NamedRule> = ctx.rules.into_iter().map(|(name, expr)| {
        let is_terminal = terminal_names.contains(&name);
        NamedRule { name, expr, is_terminal, is_internal: false }
    }).collect();

    // Mark terminal rules as internal-only when they are never referenced
    // from any nonterminal rule body.  Internal terminals exist solely as
    // sub-expressions of other terminal rules (resolved via Expr::Shared).
    let grammar_visible = collect_grammar_visible_refs(&rules, &terminal_names);
    let rules = rules.into_iter().map(|mut rule| {
        if rule.is_terminal && !grammar_visible.contains(&rule.name) {
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
    rules: Vec<(String, GrammarExpr)>,
    rule_indices: HashMap<String, usize>,
    used_rule_names: HashSet<String>,
    ref_rule_names: HashMap<String, String>,
    ref_compile_stack: HashSet<String>,
    generated_object_rule_counter: usize,
    generated_rule_counter: usize,
    expr_dedup_cache: HashMap<String, String>,
    json_string_exact_cache: HashMap<usize, String>,
    json_string_upto_cache: HashMap<usize, String>,
    shared_ap_literal_keys: BTreeSet<String>,
    shared_ap_key_colon_expr: Option<GrammarExpr>,
    shared_ap_key_rule_cache: HashMap<Vec<String>, String>,
    draft_stack: Vec<JsonSchemaDraft>,
    convert_depth: usize,
}

impl<'a> SchemaCtx<'a> {
    fn new(root: &'a Value) -> Self {
        let mut ctx = Self {
            root_schema: root,
            rules: Vec::new(),
            rule_indices: HashMap::new(),
            used_rule_names: HashSet::new(),
            ref_rule_names: HashMap::new(),
            ref_compile_stack: HashSet::new(),
            generated_object_rule_counter: 0,
            generated_rule_counter: 0,
            expr_dedup_cache: HashMap::new(),
            json_string_exact_cache: HashMap::new(),
            json_string_upto_cache: HashMap::new(),
            shared_ap_literal_keys: collect_shared_ap_literal_keys(root),
            shared_ap_key_colon_expr: None,
            shared_ap_key_rule_cache: HashMap::new(),
            draft_stack: vec![DEFAULT_JSON_SCHEMA_DRAFT],
            convert_depth: 0,
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

                let char_expr = parse_regex(JSON_STRING_CHAR_PATTERN, true);
                let expr = GrammarExpr::TerminalExpr(LexerExpr::Repeat {
                    expr: Box::new(char_expr),
                    min: count,
                    max: Some(count),
                });

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

                let char_expr = parse_regex(JSON_STRING_CHAR_PATTERN, true);
                let expr = GrammarExpr::TerminalExpr(LexerExpr::Repeat {
                    expr: Box::new(char_expr),
                    min: 0,
                    max: Some(max),
                });

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
        let open = split_open_quote();
        let close = split_close_quote();
        let colon = split_colon_space();

        self.insert_rule(JSON_STRING_CHAR_RULE, regex_expr(JSON_STRING_CHAR_PATTERN));

        // JSON_STRING_BODY_RULE: the body regex used in split paths.
        let body_regex = if close {
            JSON_STRING_BODY_ONLY_REGEX // body chars only, no closing "
        } else {
            JSON_STRING_BODY_REGEX // body chars + closing "
        };
        self.insert_rule(JSON_STRING_BODY_RULE, regex_expr(body_regex));

        // JSON_STRING_RULE: 4-way based on open × close.
        let json_string_expr = match (open, close) {
            (false, false) => regex_expr(JSON_STRING_FULL_REGEX),
            (true, false) => sequence_or_single(vec![
                literal_expr(b"\""),
                GrammarExpr::Ref(JSON_STRING_BODY_RULE.into()),
            ]),
            (false, true) => sequence_or_single(vec![
                regex_expr(JSON_STRING_OPEN_BODY_REGEX),
                literal_expr(b"\""),
            ]),
            (true, true) => sequence_or_single(vec![
                literal_expr(b"\""),
                GrammarExpr::Ref(JSON_STRING_BODY_RULE.into()),
                literal_expr(b"\""),
            ]),
        };
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

        // JSON_KEY_COLON_BODY_RULE: the key body regex.
        // If close or colon is split, body doesn't include `"\": "` suffix.
        let kc_body_regex = if close || colon {
            JSON_STRING_BODY_ONLY_REGEX
        } else {
            JSON_KEY_COLON_BODY_REGEX // body + `"\": `
        };
        self.insert_rule(JSON_KEY_COLON_BODY_RULE, regex_expr(kc_body_regex));

        // JSON_KEY_COLON_RULE: respects all split flags.
        let json_key_colon_expr = wrap_key_colon_regex(JSON_STRING_BODY_ONLY_REGEX);
        self.insert_rule(JSON_KEY_COLON_RULE, json_key_colon_expr);
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

    fn find_local_anchor_target<'v>(node: &'v Value, ref_value: &str) -> Option<&'v Value> {
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
            return Self::find_local_anchor_target(self.root_schema, ref_value)
                .ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!("unknown $ref target '{ref_value}'"))
                });
        }

        let mut current = self.root_schema;
        for token in ref_value[2..].split('/') {
            let key = Self::decode_ref_token(token);
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

    fn collect_ordered_closed_object_schema_variant(
        &mut self,
        variant: &Map<String, Value>,
    ) -> Result<Option<OrderedClosedObjectSchemaVariant>, GlrMaskError> {
        let is_object = variant.get("type").and_then(Value::as_str) == Some("object")
            || variant.contains_key("properties");
        if !is_object {
            return Ok(None);
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
        key_exprs: HashMap<String, GrammarExpr>,
        mode: StructuralBranchMode,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if variants.is_empty() || variants.len() > EXACT_CLOSED_OBJECT_UNION_MAX_VARIANTS {
            return Ok(None);
        }
        if key_exprs.len() > EXACT_CLOSED_OBJECT_UNION_MAX_KEYS {
            return Ok(None);
        }

        let start_state: Vec<OrderedSubsetCursor> = (0..variants.len())
            .map(|variant_idx| OrderedSubsetCursor {
                variant_idx: variant_idx as u16,
                cursor: 0,
            })
            .collect();
        let mut states = vec![start_state.clone()];
        let mut transitions: Vec<Vec<(String, usize)>> = vec![Vec::new()];
        let mut state_to_idx = HashMap::new();
        state_to_idx.insert(start_state, 0usize);
        let mut queue = VecDeque::from([0usize]);

        while let Some(state_idx) = queue.pop_front() {
            let state = states[state_idx].clone();
            let mut edges = Vec::new();
            for key in Self::ordered_subset_legal_next_keys(&variants, &state) {
                let next_state = Self::ordered_subset_transition(&variants, &state, &key);
                if next_state.is_empty() {
                    continue;
                }
                let next_idx = if let Some(&idx) = state_to_idx.get(&next_state) {
                    idx
                } else {
                    let idx = states.len();
                    if idx >= EXACT_CLOSED_OBJECT_UNION_MAX_STATES {
                        return Ok(None);
                    }
                    states.push(next_state.clone());
                    transitions.push(Vec::new());
                    state_to_idx.insert(next_state, idx);
                    queue.push_back(idx);
                    idx
                };
                edges.push((key, next_idx));
            }
            transitions[state_idx] = edges;
        }

        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_ord_q_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_s0")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;
        if variants.len() == 1 {
            let body_rule_names: Vec<String> = (0..states.len())
                .map(|idx| format!("{base_name}_s{idx}"))
                .collect();
            let after_rule_names: Vec<String> = (0..states.len())
                .map(|idx| format!("{base_name}_a{idx}"))
                .collect();

            for state_idx in 0..states.len() {
                let state = &states[state_idx];
                let mut alts = Vec::new();

                if Self::ordered_subset_close_allowed(mode, &variants, state) {
                    alts.push(empty_expr());
                }

                for (key, next_idx) in &transitions[state_idx] {
                    let Some(value_expr) = key_exprs.get(key).cloned() else {
                        return Ok(None);
                    };
                    let kv_expr = self.build_merged_literal_key_value_expr(b"", key, value_expr);
                    alts.push(sequence_or_single(vec![
                        kv_expr,
                        GrammarExpr::Ref(after_rule_names[*next_idx].clone()),
                    ]));
                }

                let expr = if alts.is_empty() {
                    never_expr()
                } else {
                    choice_or_single(alts)
                };
                self.insert_rule(body_rule_names[state_idx].clone(), expr);
            }

            for state_idx in 0..states.len() {
                let state = &states[state_idx];
                let mut alts = Vec::new();

                if Self::ordered_subset_close_allowed(mode, &variants, state) {
                    alts.push(empty_expr());
                }
                if !transitions[state_idx].is_empty() {
                    alts.push(sequence_or_single(vec![
                        self.json_item_separator_expr(),
                        GrammarExpr::Ref(body_rule_names[state_idx].clone()),
                    ]));
                }

                let expr = if alts.is_empty() {
                    never_expr()
                } else {
                    choice_or_single(alts)
                };
                self.insert_rule(after_rule_names[state_idx].clone(), expr);
            }

            return Ok(Some(sequence_or_single(vec![
                literal_expr(b"{"),
                GrammarExpr::Ref(body_rule_names[0].clone()),
                literal_expr(b"}"),
            ])));
        }

        let rule_names: Vec<String> = (0..states.len())
            .map(|idx| format!("{base_name}_s{idx}"))
            .collect();

        for state_idx in 0..states.len() {
            let state = &states[state_idx];
            let mut alts = Vec::new();

            if Self::ordered_subset_close_allowed(mode, &variants, state) {
                alts.push(empty_expr());
            }

            for (key, next_idx) in &transitions[state_idx] {
                let Some(value_expr) = key_exprs.get(key).cloned() else {
                    return Ok(None);
                };
                let kv_expr = self.build_merged_literal_key_value_expr(b"", key, value_expr);
                if Self::ordered_subset_close_allowed(mode, &variants, &states[*next_idx]) {
                    alts.push(kv_expr.clone());
                }
                if !transitions[*next_idx].is_empty() {
                    alts.push(sequence_or_single(vec![
                        kv_expr,
                        self.json_item_separator_expr(),
                        GrammarExpr::Ref(rule_names[*next_idx].clone()),
                    ]));
                }
            }

            let expr = if alts.is_empty() {
                never_expr()
            } else {
                choice_or_single(alts)
            };
            self.insert_rule(rule_names[state_idx].clone(), expr);
        }

        Ok(Some(sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(rule_names[0].clone()),
            literal_expr(b"}"),
        ])))
    }

    fn try_build_exact_closed_object(
        &mut self,
        ordered: &[(String, GrammarExpr, bool)],
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if ordered.is_empty() || ordered.len() > EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS {
            return Ok(None);
        }

        let key_exprs: HashMap<String, GrammarExpr> = ordered
            .iter()
            .map(|(key, value_expr, _)| (key.clone(), value_expr.clone()))
            .collect();
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
            key_exprs,
            StructuralBranchMode::AnyOf,
        )
    }

    fn try_build_exact_closed_object_union(
        &mut self,
        schema: &Map<String, Value>,
        options: &[Value],
        mode: StructuralBranchMode,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        if options.len() < 2 || options.len() > EXACT_CLOSED_OBJECT_UNION_MAX_VARIANTS {
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
            let Some(ordered) = self.collect_ordered_closed_object_schema_variant(variant)? else {
                return Ok(None);
            };
            schema_variants.push(ordered);
        }

        let mut key_schemas: HashMap<String, Value> = HashMap::new();
        let mut required_by_key: HashMap<String, bool> = HashMap::new();
        for variant in &schema_variants {
            for item in &variant.items {
                required_by_key
                    .entry(item.key.clone())
                    .and_modify(|required| *required |= item.required)
                    .or_insert(item.required);
                if let Some(existing) = key_schemas.get(&item.key) {
                    if existing != &item.value_schema {
                        return Ok(None);
                    }
                } else {
                    key_schemas.insert(item.key.clone(), item.value_schema.clone());
                }
            }
        }
        if key_schemas.len() > EXACT_CLOSED_OBJECT_UNION_MAX_KEYS {
            return Ok(None);
        }

        let mut key_exprs: HashMap<String, GrammarExpr> = HashMap::new();
        let mut skipped_optional_keys = BTreeSet::new();
        for (key, value_schema) in &key_schemas {
            match self.convert_schema(value_schema) {
                Ok(expr) => {
                    key_exprs.insert(key.clone(), expr);
                }
                Err(err) if is_unsat_schema_error(&err) => {
                    if required_by_key.get(key).copied().unwrap_or(false) {
                        return Err(unsat_schema_error());
                    }
                    skipped_optional_keys.insert(key.clone());
                }
                Err(err) => return Err(err),
            }
        }

        let variants: Vec<OrderedClosedObjectVariant> = schema_variants
            .into_iter()
            .map(|variant| OrderedClosedObjectVariant {
                items: variant
                    .items
                    .into_iter()
                    .filter(|item| !skipped_optional_keys.contains(&item.key))
                    .map(|item| OrderedClosedObjectItem {
                        value_expr: key_exprs[&item.key].clone(),
                        key: item.key,
                        required: item.required,
                    })
                    .collect(),
            })
            .collect();

        self.build_exact_ordered_closed_object_variants(variants, key_exprs, mode)
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

        // Guard 1: every variant must be a plain object with
        //          additionalProperties: false, no patternProperties.
        for v in &resolved {
            let is_object = v.get("type").and_then(Value::as_str) == Some("object")
                || v.contains_key("properties");
            if !is_object {
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

        let (tree_expr, tree_can_be_empty) = if ordered.is_empty() {
            (empty_expr(), true)
        } else {
            let mut next_idx = 0usize;
            self.build_object_tree(&base_name, &ordered, &mut next_idx)?
        };

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

        let exact_union_disabled = Self::exact_closed_object_disabled();
        if !exact_union_disabled {
            let mode = match keyword {
                "anyOf" => Some(StructuralBranchMode::AnyOf),
                "oneOf" => Some(StructuralBranchMode::OneOf),
                _ => None,
            };
            if let Some(mode) = mode {
                if let Some(expr) = self.try_build_exact_closed_object_union(schema, options, mode)? {
                    return Ok(Some(expr));
                }
            }
        }

        // Closed-object merge optimization (experimental, disabled by default).
        // Currently changes the accepted language (introduces false positives).
        // Enable with GLRMASK_MERGE_ANYOF=1 for experimentation.
        if std::env::var("GLRMASK_MERGE_ANYOF").map_or(false, |v| v == "1") {
            if let Some(merged) = self.try_merge_anyof_closed_objects(schema, options)? {
                return Ok(Some(merged));
            }
        }

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
        self.convert_depth += 1;
        let result = self.convert_schema_inner(schema);
        self.convert_depth -= 1;
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
            let base_dfa = compile_regex_union_dfa(&regexes);
            let multiple_dfa = match supported {
                SupportedMultipleOf::Integer(multiple) => {
                    integer_multiple_dfa(multiple, allow_fractional_zero)
                }
                SupportedMultipleOf::ReciprocalPowerOfTen(scale) => {
                    reciprocal_power_of_ten_dfa(scale)
                }
            };
            let intersected = Self::intersect_lexer_dfa(&base_dfa, &multiple_dfa);
            this.build_lexer_dfa_expr(&intersected, "JSON_NUMBER_MULTIPLE_OF")
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
                return Ok(json_wrapped_fullmatch_pattern(&format!("{}+", unit_pattern)));
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
                    return Ok(json_wrapped_pattern(&pattern));
                }
                // Pattern can produce strings shorter than minLength.
                // Intersect with a minimum-length regex.  Use {min,} to
                // keep the length DFA small (avoids explosion from large
                // maxLength values).
                let search_regex = string_value_body_regex(&json_search_pattern(&pattern));
                let search_dfa = build_regex(&[parse_regex(&search_regex, true)]).dfa;
                let length_inner = match max_len {
                    Some(ml) if ml <= 100 => format!(r#"(?:{}){{{},{}}}"#, JSON_STRING_CHAR_PATTERN, min_len, ml),
                    _ => format!(r#"(?:{}){{{},}}"#, JSON_STRING_CHAR_PATTERN, min_len),
                };
                let length_regex = string_value_body_regex(&length_inner);
                let length_dfa = build_regex(&[parse_regex(&length_regex, true)]).dfa;
                let intersected = Self::intersect_lexer_dfa(&search_dfa, &length_dfa);
                let body = self.build_lexer_dfa_expr(&intersected, "JSON_STRING_PATTERN_ANCHORED_BOUNDED");
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
                        let search_dfa = build_regex(&[parse_regex(&search_regex, true)]).dfa;
                        let length_regex = json_wrapped_string_length_regex(min_len, ml);
                        let length_dfa = build_regex(&[parse_regex(&length_regex, true)]).dfa;
                        let intersected = Self::intersect_lexer_dfa(&search_dfa, &length_dfa);
                        let body = self.build_lexer_dfa_expr(&intersected, "JSON_STRING_PATTERN_BOUNDED");
                        return Ok(wrap_string_value_terminal(body));
                    }
                }
                // maxLength is too large for exact intersection — apply pattern
                // with minLength enforcement only (drop maxLength to avoid
                // DFA explosion).
                if min_len > 0 {
                    let search_regex = string_value_body_regex(&json_search_pattern(&pattern));
                    let search_dfa = build_regex(&[parse_regex(&search_regex, true)]).dfa;
                    let length_inner = format!(r#"(?:{}){{{},}}"#, JSON_STRING_CHAR_PATTERN, min_len);
                    let length_regex = string_value_body_regex(&length_inner);
                    let length_dfa = build_regex(&[parse_regex(&length_regex, true)]).dfa;
                    let intersected = Self::intersect_lexer_dfa(&search_dfa, &length_dfa);
                    let body = self.build_lexer_dfa_expr(&intersected, "JSON_STRING_PATTERN_MINLEN");
                    return Ok(wrap_string_value_terminal(body));
                }
                return Ok(json_wrapped_pattern(&pattern));
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
                    regex_expr(json_hostname_label_pattern()),
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
            _ => json_format_pattern(format_name)
                .map(json_wrapped_fullmatch_pattern)
                .ok_or_else(|| GlrMaskError::GrammarParse(format!("Unknown format: {format_name}"))),
        }
    }

    fn json_literal(&self, value: &Value) -> GrammarExpr {
        json_value_literal_expr(value)
    }

    fn json_key_colon_literal(&self, text: &str) -> GrammarExpr {
        let body = literal_expr(&key_colon_literal_body_bytes(text));
        wrap_key_colon_terminal(body)
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

        let value_expr = if let Some(GrammarExpr::Literal(last)) = parts.last_mut() {
            match value_expr {
                GrammarExpr::Literal(mut bytes)
                    if matches!(bytes.first(), Some(b'{') | Some(b'[')) =>
                {
                    last.push(bytes.remove(0));
                    if bytes.is_empty() {
                        empty_expr()
                    } else {
                        literal_expr(&bytes)
                    }
                }
                GrammarExpr::Sequence(value_parts) => {
                    // Flatten nested Sequences so we can reach the leading Literal
                    let mut flat = Vec::new();
                    for part in value_parts {
                        match part {
                            GrammarExpr::Sequence(inner) => flat.extend(inner),
                            other => flat.push(other),
                        }
                    }
                    if let Some(GrammarExpr::Literal(bytes)) = flat.first_mut() {
                        if let Some(&first) = bytes.first() {
                            if matches!(first, b'{' | b'[') {
                                last.push(first);
                                bytes.remove(0);
                                if bytes.is_empty() {
                                    flat.remove(0);
                                }
                            }
                        }
                    }
                    sequence_or_single(flat)
                }
                other => other,
            }
        } else {
            value_expr
        };

        // Flatten the value into parts to avoid nested Sequences
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
                if let Some(GrammarExpr::Sequence(prev_parts)) = parts.last_mut() {
                    if let Some(GrammarExpr::Literal(prev_last)) = prev_parts.last_mut() {
                        if matches!(prev_last.last(), Some(b'}') | Some(b']')) {
                            prev_last.extend_from_slice(&leading_literal);
                            leading_literal.clear();
                        }
                    }
                } else if let Some(GrammarExpr::Literal(prev_last)) = parts.last_mut() {
                    if matches!(prev_last.last(), Some(b'}') | Some(b']')) {
                        prev_last.extend_from_slice(&leading_literal);
                        leading_literal.clear();
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
            if let Some(GrammarExpr::Sequence(last_parts)) = parts.last_mut() {
                if let Some(GrammarExpr::Literal(last_literal)) = last_parts.last_mut() {
                    if matches!(last_literal.last(), Some(b'}') | Some(b']')) {
                        last_literal.extend_from_slice(trailing_literal);
                        return sequence_or_single(parts);
                    }
                }
            } else if let Some(GrammarExpr::Literal(last_literal)) = parts.last_mut() {
                if matches!(last_literal.last(), Some(b'}') | Some(b']')) {
                    last_literal.extend_from_slice(trailing_literal);
                    return sequence_or_single(parts);
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

    fn json_item_separator_expr(&self) -> GrammarExpr {
        literal_expr(JSON_ITEM_SEPARATOR)
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

        let nonempty = sequence_or_single(vec![
            literal_expr(b"{"),
            pair.clone(),
            repeat_expr(
                sequence_or_single(vec![self.json_item_separator_expr(), pair]),
                min_pairs.saturating_sub(1),
                max_pairs.map(|max| max.saturating_sub(1)),
            ),
            literal_expr(b"}"),
        ]);

        if min_pairs == 0 {
            choice_or_single(vec![
                sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]),
                nonempty,
            ])
        } else {
            nonempty
        }
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

    fn scoped_key_colon_dfa(property_names: Option<&Value>) -> Result<LexerDfa, GlrMaskError> {
        let pattern = if let Some(property_names) = property_names {
            key_colon_body_regex(
                &json_search_pattern(Self::property_name_pattern(property_names)?),
            )
        } else {
            key_colon_body_regex(JSON_STRING_BODY_ONLY_REGEX)
        };
        Ok(build_regex(&[parse_regex(&pattern, true)]).dfa)
    }

    fn pattern_key_colon_dfa(pattern: &str) -> LexerDfa {
        let inner = json_search_pattern(pattern);
        let pat = key_colon_body_regex(&inner);
        build_regex(&[parse_regex(&pat, true)]).dfa
    }

    fn literal_key_colon_union_dfa(keys: &BTreeSet<String>) -> Option<LexerDfa> {
        if keys.is_empty() {
            return None;
        }
        let exprs = keys
            .iter()
            .map(|key| LexerExpr::U8Seq(key_colon_literal_body_bytes(key)))
            .collect::<Vec<_>>();
        let expr = if exprs.len() == 1 {
            exprs.into_iter().next().unwrap()
        } else {
            LexerExpr::Choice(exprs)
        };
        Some(build_regex(&[expr]).dfa)
    }

    fn dfa_accepts_any(dfa: &LexerDfa) -> bool {
        dfa.states().iter().any(|state| !state.finalizers.is_empty())
    }

    fn build_product_lexer_dfa<State, IsAccepting, Transitions>(
        start: State,
        mut is_accepting: IsAccepting,
        mut transitions_for: Transitions,
    ) -> LexerDfa
    where
        State: Copy + Eq + std::hash::Hash,
        IsAccepting: FnMut(State) -> bool,
        Transitions: FnMut(State) -> Vec<(u8, State)>,
    {
        let mut state_ids = HashMap::<State, usize>::new();
        let mut worklist = VecDeque::<State>::new();
        let mut transitions = Vec::<Vec<(u8, u32)>>::new();
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
                entries.push((byte, next_result_state_id as u32));
            }
            transitions[result_state_id] = entries;
        }

        let mut dfa = LexerDfa::new(transitions.len());
        dfa.ensure_group_capacity(1);
        for (state_id, entries) in transitions.into_iter().enumerate() {
            dfa.set_transitions_from_sorted_entries(state_id as u32, entries);
            let mut finalizers = BitSet::new(1);
            if accepting[state_id] {
                finalizers.set(0);
            }
            dfa.overwrite_state_metadata(state_id as u32, finalizers, BitSet::new(1));
        }
        let start_u8set = dfa.get_u8set(0);
        dfa.set_group_u8set(0, start_u8set);
        dfa.minimize()
    }

    fn intersect_lexer_dfa(left: &LexerDfa, right: &LexerDfa) -> LexerDfa {
        Self::build_product_lexer_dfa(
            (0u32, 0u32),
            |(left_state_id, right_state_id)| {
                !left.states()[left_state_id as usize].finalizers.is_empty()
                    && !right.states()[right_state_id as usize].finalizers.is_empty()
            },
            |(left_state_id, right_state_id)| {
                left.states()[left_state_id as usize]
                    .transitions
                    .iter()
                    .filter_map(|(byte, &left_next)| {
                        right
                            .step(right_state_id, byte)
                            .map(|right_next| (byte, (left_next, right_next)))
                    })
                    .collect()
            },
        )
    }

    fn build_lexer_dfa_expr(&mut self, dfa: &LexerDfa, prefix: &str) -> GrammarExpr {
        if !Self::dfa_accepts_any(dfa) {
            return never_expr();
        }

        self.extract_terminal_rule(
            GrammarExpr::TerminalExpr(LexerExpr::Dfa(dfa.clone())),
            prefix,
        )
    }

    /// Build a grammar expression for a key-colon DFA, wrapping it with
    /// the split-off literal parts (opening quote, close quote, colon-space)
    /// just like `wrap_key_colon_regex` does for regex-based keys.
    fn build_key_colon_dfa_expr(&mut self, dfa: &LexerDfa, prefix: &str) -> GrammarExpr {
        self.build_excluding_key_colon_dfa_expr(dfa, &[], prefix)
    }

    /// Build a key-colon terminal from a base DFA, excluding any keys matched
    /// by the `excluded` DFAs.  Uses `GrammarExpr::Exclude` instead of
    /// DFA-level subtraction so that the grammar compiler handles exclusion.
    fn build_excluding_key_colon_dfa_expr(
        &mut self,
        base_dfa: &LexerDfa,
        excluded_dfas: &[&LexerDfa],
        prefix: &str,
    ) -> GrammarExpr {
        if !Self::dfa_accepts_any(base_dfa) {
            return never_expr();
        }
        let base_body = GrammarExpr::TerminalExpr(LexerExpr::Dfa(base_dfa.clone()));
        let body = if excluded_dfas.is_empty() {
            base_body
        } else {
            let excluded_exprs: Vec<GrammarExpr> = excluded_dfas
                .iter()
                .map(|dfa| GrammarExpr::TerminalExpr(LexerExpr::Dfa((*dfa).clone())))
                .collect();
            GrammarExpr::Exclude {
                expr: Box::new(base_body),
                exclude: Box::new(choice_or_single(excluded_exprs)),
            }
        };
        let body_terminal = self.extract_terminal_rule(body, prefix);
        wrap_key_colon_terminal(body_terminal)
    }

    fn build_excluding_key_colon_expr(
        &mut self,
        base_key_colon_body_expr: GrammarExpr,
        excluded_key_colon_body_exprs: Vec<GrammarExpr>,
        prefix: &str,
    ) -> GrammarExpr {
        let expr = if excluded_key_colon_body_exprs.is_empty() {
            base_key_colon_body_expr
        } else {
            GrammarExpr::Exclude {
                expr: Box::new(base_key_colon_body_expr),
                exclude: Box::new(choice_or_single(excluded_key_colon_body_exprs)),
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
            let key_dfa = Self::scoped_key_colon_dfa(None)?;
            let shared_excluded_dfa = Self::literal_key_colon_union_dfa(&self.shared_ap_literal_keys);
            let mut excluded_dfas: Vec<&LexerDfa> = Vec::new();
            if let Some(ref dfa) = shared_excluded_dfa {
                excluded_dfas.push(dfa);
            }
            self.build_excluding_key_colon_dfa_expr(
                &key_dfa,
                &excluded_dfas,
                "AP_SHARED_KEY_COLON",
            )
        };

        self.shared_ap_key_colon_expr = Some(expr.clone());
        Ok(expr)
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
            let additional_schema =
                self.normalized_additional_properties_schema(additional_properties);

            return self.build_ordered_properties_object_expr(
                properties,
                &required_list,
                &required_keys,
                &pattern_property_entries,
                additional_schema,
                property_names,
            );
        }

        let additional_schema = self.normalized_additional_properties_schema(additional_properties);

        if property_names.is_none() && pattern_properties.is_none() && (min_properties.is_some() || max_properties.is_some()) {
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

        let extra_pair = if let Some(schema) = additional_properties_schema {
            let extra_value_expr = self.convert_schema(&schema)?;
            if ap_key_any_string() {
                // Skip key exclusion — AP keys accept any JSON string.
                // Each object still gets its own uniquely-named terminal.
                let full_key_dfa = Self::scoped_key_colon_dfa(None)?;
                Some(sequence_or_single(vec![
                    self.build_key_colon_dfa_expr(
                        &full_key_dfa,
                        &format!("{}_KEY_COLON", base_name.to_uppercase()),
                    ),
                    extra_value_expr,
                ]))
            } else if shared_ap_key_exclusions_enabled() {
                let mut excluded_literal_keys = BTreeSet::<String>::new();
                excluded_literal_keys.extend(properties.keys().cloned());
                excluded_literal_keys.extend(required_list.iter().cloned());
                Some(sequence_or_single(vec![
                    self.build_shared_additional_key_choice_expr(
                        &excluded_literal_keys,
                        &format!("{base_name}_ap_key"),
                    )?,
                    extra_value_expr,
                ]))
            } else {
                let mut excluded_literal_keys = BTreeSet::<String>::new();
                excluded_literal_keys.extend(properties.keys().cloned());
                excluded_literal_keys.extend(required_list.iter().cloned());
                let key_dfa = Self::scoped_key_colon_dfa(None)?;
                let mut excluded_dfas: Vec<&LexerDfa> = Vec::new();
                let excluded_literal_dfa = Self::literal_key_colon_union_dfa(&excluded_literal_keys);
                if let Some(ref dfa) = excluded_literal_dfa {
                    excluded_dfas.push(dfa);
                }
                Some(sequence_or_single(vec![
                    self.build_excluding_key_colon_dfa_expr(
                        &key_dfa,
                        &excluded_dfas,
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

    fn build_array_item_sequence(
        &mut self,
        items: &[(GrammarExpr, bool)],
        needs_separator: bool,
        cache: &mut HashMap<(usize, bool), GrammarExpr>,
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
        let supports_at_least_one = residual_min == 1 && residual_max.is_none();
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

        for optional_key in &optional_keys {
            let option_required = required_list
                .iter()
                .cloned()
                .chain(std::iter::once(optional_key.clone()))
                .collect::<Vec<_>>();
            let option_required_keys = option_required.iter().cloned().collect::<BTreeSet<_>>();
            options.push(self.build_ordered_properties_object_expr(
                properties,
                &option_required,
                &option_required_keys,
                &empty_patterns,
                None,
                property_names,
            )?);
        }
        Ok(Some(choice_or_single(options)))
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

        // Additional properties are emitted only after the fixed object tree.
        // Exclude all declared keys (both required and optional) so that a key
        // already handled by the tree cannot also appear as an additional property.
        let additional_excluded_literal_keys = fixed_literal_keys.clone();

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

        let mut base_index = self.generated_object_rule_counter;
        let base_name = loop {
            let candidate = format!("obj_ord_{base_index}");
            if !self.used_rule_names.contains(&format!("{candidate}_0_nc")) {
                break candidate;
            }
            base_index += 1;
        };
        self.generated_object_rule_counter = base_index + 1;

        let key_dfa_needed = !pattern_properties.is_empty()
            || property_names.is_some()
            || !additional_excluded_literal_keys.is_empty();
        let (base_key_colon_dfa, fixed_key_union_dfa, additional_fixed_key_union_dfa, pattern_key_colon_dfas) =
            if key_dfa_needed {
                (
                    Some(Self::scoped_key_colon_dfa(property_names)?),
                    Self::literal_key_colon_union_dfa(&fixed_literal_keys),
                    Self::literal_key_colon_union_dfa(&additional_excluded_literal_keys),
                    pattern_properties
                        .iter()
                        .map(|(pattern, _)| Self::pattern_key_colon_dfa(pattern))
                        .collect::<Vec<_>>(),
                )
            } else {
                (None, None, None, Vec::new())
            };

        let mut pattern_pair_exprs = Vec::<GrammarExpr>::new();

        if !pattern_properties.is_empty() {
            let mut subsets = Vec::<Vec<usize>>::new();
            let mut current = Vec::<usize>::new();
            Self::collect_nonempty_index_subsets(
                pattern_properties.len(),
                0,
                &mut current,
                &mut subsets,
            );

            for (subset_idx, subset) in subsets.into_iter().enumerate() {
                let mut key_dfa = base_key_colon_dfa
                    .as_ref()
                    .expect("pattern/property-name DFA should exist when needed")
                    .clone();
                for pattern_idx in &subset {
                    key_dfa = Self::intersect_lexer_dfa(&key_dfa, &pattern_key_colon_dfas[*pattern_idx]);
                    if !Self::dfa_accepts_any(&key_dfa) {
                        break;
                    }
                }
                if !Self::dfa_accepts_any(&key_dfa) {
                    continue;
                }

                // Collect DFAs to exclude via GrammarExpr::Exclude
                let subset_members: HashSet<usize> = subset.iter().copied().collect();
                let mut excluded_dfas: Vec<&LexerDfa> = Vec::new();
                if let Some(ref fk_dfa) = fixed_key_union_dfa {
                    excluded_dfas.push(fk_dfa);
                }
                for (pattern_idx, pattern_dfa) in pattern_key_colon_dfas.iter().enumerate() {
                    if !subset_members.contains(&pattern_idx) {
                        excluded_dfas.push(pattern_dfa);
                    }
                }

                let key_expr = self.build_excluding_key_colon_dfa_expr(
                    &key_dfa,
                    &excluded_dfas,
                    &format!("{}_PP_SUBSET_{}_KEY", base_name.to_uppercase(), subset_idx),
                );

                let subset_schemas = subset
                    .iter()
                    .map(|idx| pattern_properties[*idx].1.clone())
                    .collect::<Vec<_>>();
                let subset_schema = Self::schema_all_of(subset_schemas);

                let value_expr = match self.convert_schema(&subset_schema) {
                    Ok(expr) => expr,
                    Err(err) if is_unsat_schema_error(&err) => continue,
                    Err(err) => return Err(err),
                };

                pattern_pair_exprs.push(sequence_or_single(vec![key_expr, value_expr]));
            }
        }

        let mut additional_pair_exprs = Vec::<GrammarExpr>::new();
        let has_additional_properties = additional_properties_schema.is_some();
        if let Some(schema) = additional_properties_schema {
            let additional_key_expr = if ap_key_any_string() {
                // Skip key exclusion — AP keys accept any JSON string.
                // Each object still gets its own uniquely-named terminal to preserve grammar structure.
                let full_key_dfa = Self::scoped_key_colon_dfa(property_names)?;
                Some(self.build_key_colon_dfa_expr(
                    &full_key_dfa,
                    &format!("{}_AP_KEY", base_name.to_uppercase()),
                ))
            } else if shared_ap_key_exclusions_enabled()
                && pattern_properties.is_empty()
                && property_names.is_none()
            {
                Some(self.build_shared_additional_key_choice_expr(
                    &additional_excluded_literal_keys,
                    &format!("{base_name}_ap_key"),
                )?)
            } else if key_dfa_needed {
                let additional_key_dfa = base_key_colon_dfa
                    .as_ref()
                    .expect("additional-properties DFA should exist when needed");
                let mut excluded_dfas: Vec<&LexerDfa> = Vec::new();
                if let Some(ref fk_dfa) = additional_fixed_key_union_dfa {
                    excluded_dfas.push(fk_dfa);
                }
                for pattern_dfa in &pattern_key_colon_dfas {
                    excluded_dfas.push(pattern_dfa);
                }
                Some(self.build_excluding_key_colon_dfa_expr(
                    additional_key_dfa,
                    &excluded_dfas,
                    &format!("{}_AP_KEY", base_name.to_uppercase()),
                ))
            } else {
                Some(self.json_key_colon_ref())
            };

            if let Some(additional_key_expr) = additional_key_expr {
                let additional_value_expr = self.convert_schema(&schema)?;
                additional_pair_exprs.push(sequence_or_single(vec![
                    additional_key_expr,
                    additional_value_expr,
                ]));
            }
        }

        // Keep all properties (required AND optional) in the ordered tree so
        // that property ordering is always enforced.  The GLR parser handles
        // the resulting LR(2) ambiguity (`, ` after an optional tree leaf could
        // continue to the next tree property OR to the continuation chain) by
        // forking and pruning dead paths automatically.
        //
        // Previously, optional properties were moved to the unordered "free"
        // continuation chain when additionalProperties or patternProperties
        // existed, which destroyed property ordering and caused false-accepts.
        let _has_continuation = !additional_pair_exprs.is_empty() || !pattern_pair_exprs.is_empty();

        let (additional_nc, additional_c) = self.build_object_pair_tail(
            &base_name,
            "ap",
            additional_pair_exprs,
            empty_expr(),
            empty_expr(),
        );
        let free_pair_exprs = pattern_pair_exprs;
        let (free_nc, free_c) = self.build_object_pair_tail(
            &base_name,
            "free",
            free_pair_exprs,
            GrammarExpr::Ref(additional_nc.clone()),
            GrammarExpr::Ref(additional_c.clone()),
        );

        if !Self::exact_closed_object_disabled()
            && !ordered.is_empty()
            && ordered.iter().any(|(_, _, required)| !*required)
            && pattern_properties.is_empty()
            && !has_additional_properties
            && property_names.is_none()
        {
            if let Some(expr) = self.try_build_exact_closed_object(&ordered)? {
                return Ok(expr);
            }
        }

        if !ordered.is_empty()
            && ordered.iter().all(|(_, _, required)| *required)
            && pattern_properties.is_empty()
            && !has_additional_properties
        {
            return Ok(self.build_closed_required_ordered_object_expr(&ordered));
        }

        if !ordered.is_empty() && ordered.iter().all(|(_, _, required)| *required) {
            let prefix = self.build_required_ordered_object_body_expr(&ordered, None);
            return Ok(sequence_or_single(vec![
                prefix,
                GrammarExpr::Ref(free_c.clone()),
                literal_expr(b"}"),
            ]));
        }

        if ordered.is_empty() {
            return Ok(sequence_or_single(vec![
                literal_expr(b"{"),
                GrammarExpr::Ref(free_nc),
                literal_expr(b"}"),
            ]));
        }

        let mut next_tree_rule_index = 0usize;
        let (tree_expr, tree_can_be_empty) =
            self.build_object_tree(&base_name, &ordered, &mut next_tree_rule_index)?;
        let tree_prefix = sequence_or_single(vec![literal_expr(b"{"), tree_expr]);
        let with_tree_prefix = sequence_or_single(vec![
            tree_prefix,
            GrammarExpr::Ref(free_c.clone()),
            literal_expr(b"}"),
        ]);

        if tree_can_be_empty {
            return Ok(choice_or_single(vec![
                with_tree_prefix,
                sequence_or_single(vec![
                    literal_expr(b"{"),
                    GrammarExpr::Ref(free_nc),
                    literal_expr(b"}"),
                ]),
            ]));
        }

        Ok(with_tree_prefix)
    }

    fn build_object_tree(
        &mut self,
        base_name: &str,
        items: &[(String, GrammarExpr, bool)],
        next_rule_index: &mut usize,
    ) -> Result<(GrammarExpr, bool), GlrMaskError> {
        if items.len() == 1 {
            let (key, value_expr, is_required) = &items[0];
            let kv_expr = self.build_merged_literal_key_value_expr(b"", key, value_expr.clone());
            if *is_required {
                return Ok((kv_expr, false));
            }

            let rule_name = format!("{base_name}_t{}", *next_rule_index);
            *next_rule_index += 1;
            self.insert_rule(rule_name.clone(), kv_expr);
            return Ok((GrammarExpr::Ref(rule_name), true));
        }

        let mid = items.len() / 2;
        let (left_expr, left_can_be_empty) =
            self.build_object_tree(base_name, &items[..mid], next_rule_index)?;
        let (right_expr, right_can_be_empty) =
            self.build_object_tree(base_name, &items[mid..], next_rule_index)?;

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

        let rule_name = format!("{base_name}_t{}", *next_rule_index);
        *next_rule_index += 1;
        self.insert_rule(rule_name.clone(), choice_or_single(options));
        Ok((GrammarExpr::Ref(rule_name), left_can_be_empty && right_can_be_empty))
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
        let pair = sequence_or_single(vec![json_wrapped_key_colon_pattern(pattern), value_expr]);
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
        let matched_key_colon_body = regex_expr(key_colon_body_regex(&json_search_pattern(pattern)));
        let matched_key_colon_full = wrap_key_colon_terminal(matched_key_colon_body.clone());
        let matched_pair = sequence_or_single(vec![
            matched_key_colon_full,
            matched_value_expr,
        ]);
        let unmatched_pair = sequence_or_single(vec![
            self.build_excluding_key_colon_expr(
                unmatched_key_colon_body,
                vec![matched_key_colon_body],
                "PP_KEY_COLON",
            ),
            unmatched_value_expr,
        ]);
        let base_name = self.fresh_rule_name("obj_pat_mix");
        let (additional_nc, additional_c) = self.build_object_pair_tail(
            &base_name,
            "ap",
            vec![unmatched_pair],
            empty_expr(),
            empty_expr(),
        );
        let (matched_nc, _) = self.build_object_pair_tail(
            &base_name,
            "pp",
            vec![matched_pair],
            GrammarExpr::Ref(additional_nc),
            GrammarExpr::Ref(additional_c),
        );
        Ok(sequence_or_single(vec![
            literal_expr(b"{"),
            GrammarExpr::Ref(matched_nc),
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

            let mut sequence_cache = HashMap::new();
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

/// Collect terminal names that are referenced from nonterminal rule bodies.
/// These terminals are "grammar-visible" and must have their own TerminalID.
fn collect_grammar_visible_refs(
    rules: &[NamedRule],
    terminal_names: &HashSet<String>,
) -> HashSet<String> {
    fn walk(expr: &GrammarExpr, terminal_names: &HashSet<String>, out: &mut HashSet<String>) {
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
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::TerminalExpr(_)
            | GrammarExpr::AnyByte => {}
        }
    }
    let mut visible = HashSet::new();
    for rule in rules {
        if !rule.is_terminal {
            walk(&rule.expr, terminal_names, &mut visible);
        }
    }
    visible
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

    fn expr_has_literal_prefix(expr: &GrammarExpr, prefix: &[u8]) -> bool {
        match expr {
            GrammarExpr::Literal(bytes) => bytes.starts_with(prefix),
            GrammarExpr::Sequence(parts) => parts.iter().any(|part| expr_has_literal_prefix(part, prefix)),
            GrammarExpr::Choice(options) => options.iter().any(|part| expr_has_literal_prefix(part, prefix)),
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner)
            | GrammarExpr::RepeatRange { expr: inner, .. } => expr_has_literal_prefix(inner, prefix),
            _ => false,
        }
    }

    fn named_grammar_has_literal(grammar: &NamedGrammar, expected: &[u8]) -> bool {
        grammar.rules.iter().any(|rule| expr_has_literal(&rule.expr, expected))
    }

    fn named_grammar_has_literal_prefix(grammar: &NamedGrammar, prefix: &[u8]) -> bool {
        grammar
            .rules
            .iter()
            .any(|rule| expr_has_literal_prefix(&rule.expr, prefix))
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
    fn test_prefix_items_follow_llguidance_optional_tuple_semantics() {
        let schema = r#"{
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}]
        }"#;
        assert!(accepts_sequence(schema, &[b"[]"]));
        assert!(accepts_sequence(schema, &[b"[\"x\"]"]));
        assert!(!accepts_sequence(schema, &[b"[1]"]));
        assert!(!accepts_sequence(schema, &[b"[\"x\", \"y\"]"]));
    }

    #[test]
    fn test_items_false_matches_llguidance_array_closure() {
        let schema = r#"{
            "type": "array",
            "items": false
        }"#;
        assert!(accepts_sequence(schema, &[b"[]"]));
        assert!(!accepts_sequence(schema, &[b"[0]"]));
    }

    #[test]
    fn test_old_draft_items_array_uses_additional_items() {
        let schema = r#"{
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "array",
            "items": [{"type": "string"}, {"type": "integer"}],
            "additionalItems": false
        }"#;
        assert!(accepts_sequence(schema, &[b"[]"]));
        assert!(accepts_sequence(schema, &[b"[\"x\", 1]"]));
        assert!(!accepts_sequence(schema, &[b"[\"x\", 1, true]"]));
    }

    #[test]
    fn test_draft4_boolean_exclusive_bounds_match_llguidance() {
        let schema = r#"{
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "number",
            "minimum": 0,
            "exclusiveMinimum": true
        }"#;
        assert!(!accepts_sequence(schema, &[b"0"]));
        assert!(accepts_sequence(schema, &[b"0.1"]));
    }

    #[test]
    fn test_min_max_properties_special_cases_match_llguidance() {
        let exactly_one = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": false
        }"#;
        assert!(accepts_sequence(exactly_one, &[b"{\"a\": \"x\"}"]));
        assert!(!accepts_sequence(exactly_one, &[b"{}"]));
        assert!(!accepts_sequence(exactly_one, &[b"{\"a\": \"x\", \"b\": \"y\"}"]));

        let at_least_one = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "minProperties": 1,
            "additionalProperties": false
        }"#;
        assert!(!accepts_sequence(at_least_one, &[b"{}"]));
        assert!(accepts_sequence(at_least_one, &[b"{\"a\": \"x\", \"b\": \"y\"}"]));
    }

    #[test]
    fn test_min_properties_with_additional_properties_schema_requires_nonempty_object() {
        let schema = r#"{
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"type": "string"}
        }"#;

        assert!(!accepts_sequence(schema, &[b"{}"]));
        assert!(accepts_sequence(schema, &[b"{\"k\": \"v\"}"]));
        assert!(!accepts_sequence(schema, &[b"{\"k\": 1}"]));
    }

    #[test]
    fn test_max_properties_with_additional_properties_schema_bounds_pair_count() {
        let schema = r#"{
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "additionalProperties": {"type": "string"}
        }"#;

        assert!(!accepts_sequence(schema, &[b"{}"]));
        assert!(accepts_sequence(schema, &[b"{\"k\": \"v\"}"]));
        assert!(!accepts_sequence(schema, &[b"{\"a\": \"x\", \"b\": \"y\"}"]));
    }

    #[test]
    fn test_oneof_schema() {
        let schema = r#"{
            "oneOf": [{"type": "string"}, {"type": "integer"}]
        }"#;
        assert!(accepts_sequence(schema, &[b"\"value\""]));
        assert!(accepts_sequence(schema, &[b"17"]));
        assert!(!accepts_sequence(schema, &[b"true"]));
    }

    #[test]
    fn test_ref_oneof_does_not_widen_to_json_value() {
        let schema = r##"{
            "$defs": {
                "entry": {
                    "additionalProperties": false,
                    "properties": {
                        "description": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "type": {"enum": ["int", "str"]}
                    }
                }
            },
            "type": "object",
            "properties": {
                "argument_specs": {
                    "additionalProperties": {"$ref": "#/$defs/entry"}
                }
            },
            "additionalProperties": false
        }"##;

        assert!(accepts_sequence(
            schema,
            &[
                b"{",
                b"\"argument_specs\"",
                b": ",
                b"{",
                b"\"main\"",
                b": ",
                b"{",
                b"\"type\"",
                b": ",
                b"\"int\"",
                b"}",
                b"}",
                b"}",
            ],
        ));
        assert!(!accepts_sequence(
            schema,
            &[
                b"{",
                b"\"argument_specs\"",
                b": ",
                b"{",
                b"\"main\"",
                b": ",
                b"{",
                b"\"type lure\"",
                b": ",
                b"0",
                b"}",
                b"}",
                b"}",
            ],
        ));
        assert!(!accepts_sequence(
            schema,
            &[
                b"{",
                b"\"argument_specs\"",
                b": ",
                b"{",
                b"\"main\"",
                b": ",
                b"{",
                b"\"type\"",
                b": ",
                b"\"int Garcia\"",
                b"}",
                b"}",
                b"}",
            ],
        ));
    }

    #[test]
    fn test_ref_compile_errors_do_not_fallback_to_json_value() {
        assert_schema_error_contains(
            r##"{
                "$defs": {
                    "bad": {
                        "type": "array",
                        "uniqueItems": true
                    }
                },
                "type": "object",
                "properties": {
                    "items": {"$ref": "#/$defs/bad"}
                }
            }"##,
            "Unimplemented keys",
        );
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
        // After opening-quote separation, the start rule is a sequence:
        // [literal_expr(b"\""), Ref(JSON_STRING_BOUNDED_xxx)]
        // Verify the body terminal exists in the sequence.
        fn has_terminal_ref(expr: &GrammarExpr, grammar: &NamedGrammar) -> bool {
            match expr {
                GrammarExpr::Ref(name) => {
                    grammar.rules.iter().any(|r| r.name == *name && r.is_terminal)
                }
                GrammarExpr::Sequence(parts) => parts.iter().any(|p| has_terminal_ref(p, grammar)),
                _ => false,
            }
        }
        assert!(has_terminal_ref(&start_rule.expr, &grammar),
            "expected bounded string to contain a terminal ref for the body");
    }

    #[test]
    fn test_bounded_string_does_not_emit_exact_upto_ladders() {
        let schema: Value = serde_json::from_str(r#"{"type": "string", "minLength": 1, "maxLength": 5}"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();

        assert!(grammar.rules.iter().all(|rule| {
            !rule.name.starts_with("JSON_STRING_CHAR_EXACT_")
                && !rule.name.starts_with("JSON_STRING_CHAR_UPTO_")
        }));
    }

    #[test]
    fn test_large_bounded_string_uses_split_nonterminal_and_terminal_chunks() {
        let schema: Value = serde_json::from_str(r#"{"type": "string", "maxLength": 1025}"#).unwrap();
        let grammar = schema_to_named_grammar(&schema).unwrap();

        let start_rule = grammar.rules.iter().find(|rule| rule.name == "start").unwrap();
        let split_rule_name = match &start_rule.expr {
            GrammarExpr::Ref(rule_name) => rule_name,
            other => panic!("expected split string start to be a rule ref, got {other:?}"),
        };
        let split_rule = grammar
            .rules
            .iter()
            .find(|rule| rule.name == *split_rule_name)
            .unwrap();
        assert!(!split_rule.is_terminal, "expected large bounded string to lower through a nonterminal rule");

        let chunk = json_string_repeat_chunk();
        let exact_prefix = format!("JSON_STRING_CHAR_EXACT_{chunk}");
        let upto_prefix = format!("JSON_STRING_CHAR_UPTO_{chunk}");
        assert!(grammar.rules.iter().any(|rule| {
            rule.is_terminal && rule.name.starts_with(&exact_prefix)
        }), "expected a terminal with prefix {exact_prefix}");
        assert!(grammar.rules.iter().any(|rule| {
            rule.is_terminal && rule.name.starts_with(&upto_prefix)
        }), "expected a terminal with prefix {upto_prefix}");
    }

    #[test]
    fn test_accepts_large_split_bounded_string_length() {
        let token = format!("\"{}\"", "a".repeat(1025)).into_bytes();
        assert!(accepts_sequence(
            r#"{"type": "string", "maxLength": 1025}"#,
            &[token.as_slice()]
        ));
    }

    #[test]
    fn test_rejects_too_long_large_split_bounded_string_length() {
        let token = format!("\"{}\"", "a".repeat(1026)).into_bytes();
        assert!(!accepts_sequence(
            r#"{"type": "string", "maxLength": 1025}"#,
            &[token.as_slice()]
        ));
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
    fn test_object_typed_property_rejects_non_object_value() {
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
    fn test_bare_object_schema_rejects_non_objects() {
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
                    }
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
    fn test_exact_pattern_property_is_not_bypassed_by_additional_properties() {
        let schema = r#"{
            "type": "object",
            "patternProperties": {
                "^flag$": {
                    "type": "string",
                    "enum": ["ok"]
                }
            }
        }"#;
        assert!(accepts_sequence(schema, &[b"{\"flag\": \"ok\"}"]));
        assert!(!accepts_sequence(schema, &[b"{\"flag\": \"nope\"}"]));
        assert!(accepts_sequence(schema, &[b"{\"any\": \"thing\"}"]));
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
        // With colon-space splitting (default), the key body includes the close
        // quote but the colon-space suffix is a separate literal:
        // literal_expr(b"\"") + literal_expr(b"opacity\"") + literal_expr(b": ")
        assert!(named_grammar_has_literal(&grammar, b"opacity\""));
        assert!(named_grammar_has_literal(&grammar, b": "));
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
        // With colon-space splitting (default), the ": " suffix appears as a
        // separate literal in the grammar — that is expected.
        assert!(named_grammar_has_literal(&grammar, b": "));
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
    fn test_closed_required_object_merges_outer_literals_into_key_prefixes() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "status": {"enum": ["ok", "fail"]},
                "kind": {"enum": ["A", "B"]}
            },
            "required": ["status", "kind"],
            "additionalProperties": false
        }"#;
        let grammar = schema_to_named_grammar(&serde_json::from_str(schema).unwrap()).unwrap();

        assert!(named_grammar_has_literal(
            &grammar,
            b"{\"status\": \"ok\", \"kind\": \"A\"}"
        ));
        assert!(accepts_sequence(schema, &[b"{\"status\": \"ok\", \"kind\": \"A\"}"]));
    }

    #[test]
    fn test_nested_closed_object_merges_open_brace_into_parent_key_suffix() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "role": {"enum": ["admin", "guest"]},
                        "tier": {"enum": ["free", "pro"]}
                    },
                    "required": ["role", "tier"],
                    "additionalProperties": false
                }
            },
            "required": ["user"],
            "additionalProperties": false
        }"#;
        let grammar = schema_to_named_grammar(&serde_json::from_str(schema).unwrap()).unwrap();

        assert!(named_grammar_has_literal(
            &grammar,
            b"{\"user\": {\"role\": \"admin\", \"tier\": \"pro\"}}"
        ));
        assert!(accepts_sequence(
            schema,
            &[b"{\"user\": {\"role\": \"admin\", \"tier\": \"pro\"}}"],
        ));
    }

    #[test]
    fn test_required_open_object_still_accepts_finite_required_prefix() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "status": {"enum": ["ok", "fail"]},
                "kind": {"enum": ["A", "B"]}
            },
            "required": ["status", "kind"]
        }"#;
        assert!(accepts_sequence(
            schema,
            &[b"{\"status\": \"ok\", \"kind\": \"A\", \"extra\": 1}"],
        ));
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
        // With colon-space splitting (default), the ": " suffix is a
        // separate literal in the grammar.
        assert!(named_grammar_has_literal(&grammar, b": "));
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
        // With colon-space splitting (default), ": " is a separate literal.
        assert!(named_grammar_has_literal(&grammar, b": "));
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
    fn test_pattern_properties_take_precedence_over_additional_properties() {
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
        assert!(accepts_sequence(schema, &[b"{\"mode1\": [\"a\"]}"]));
        assert!(!accepts_sequence(schema, &[b"{\"mode1\": \"a\"}"]));
        assert!(accepts_sequence(schema, &[b"{\"other\": \"a\"}"]));
        assert!(!accepts_sequence(schema, &[b"{\"other\": [\"a\"]}"]));
    }

    #[test]
    fn test_properties_and_pattern_properties_do_not_fall_through_to_additional_properties() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "default": {"type": "string"}
            },
            "patternProperties": {
                "^mode": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "additionalProperties": {"type": "string"}
        }"#;
        assert!(accepts_sequence(
            schema,
            &[b"{\"default\": \"x\", \"mode1\": [\"a\"], \"other\": \"y\"}"]
        ));
        assert!(!accepts_sequence(
            schema,
            &[b"{\"default\": \"x\", \"mode1\": \"y\"}"]
        ));
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
    fn test_number_type_accepts_integer_and_fractional_literals() {
        let schema = r#"{"type": "number"}"#;
        assert!(accepts_sequence(schema, &[b"1"]));
        assert!(accepts_sequence(schema, &[b"1.5"]));
    }

    #[test]
    fn test_number_type_accepts_exponent_literals() {
        let schema = r#"{"type": "number"}"#;
        assert!(accepts_sequence(schema, &[b"1e1"]));
        assert!(accepts_sequence(schema, &[b"1.5e1"]));
    }

    #[test]
    fn test_integer_number_type_array_accepts_integer_and_fractional_literals() {
        let schema = r#"{"type": ["integer", "number"]}"#;
        assert!(accepts_sequence(schema, &[b"1"]));
        assert!(accepts_sequence(schema, &[b"1.5"]));
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
    fn test_integer_multiple_of_rejects_invalid_open_values() {
        let schema = r#"{
            "type": "integer",
            "multipleOf": 5
        }"#;

        assert!(accepts_sequence(schema, &[b"5"]));
        assert!(accepts_sequence(schema, &[b"10"]));
        assert!(!accepts_sequence(schema, &[b"7"]));
        assert!(!accepts_sequence(schema, &[b"12"]));
    }

    #[test]
    fn test_integer_multiple_of_rejects_invalid_bounded_values() {
        let schema = r#"{
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "multipleOf": 5
        }"#;

        assert!(accepts_sequence(schema, &[b"0"]));
        assert!(accepts_sequence(schema, &[b"5"]));
        assert!(accepts_sequence(schema, &[b"10"]));
        assert!(!accepts_sequence(schema, &[b"7"]));
    }

    #[test]
    fn test_number_integer_multiple_of_allows_zero_fraction_only() {
        let schema = r#"{
            "type": "number",
            "multipleOf": 5
        }"#;

        assert!(lexer_dfa_accepts(&integer_multiple_dfa(5, true), b"10.0"));
        assert!(accepts_sequence(schema, &[b"10"]));
        assert!(accepts_sequence(schema, &[b"10.0"]));
        assert!(accepts_sequence(schema, &[b"10.000"]));
        assert!(!accepts_sequence(schema, &[b"10.5"]));
        assert!(!accepts_sequence(schema, &[b"7"]));
    }

    #[test]
    fn test_nonnegative_power_of_ten_multiple_of_rejects_extra_precision() {
        let schema = r#"{
            "type": "number",
            "minimum": 0,
            "multipleOf": 0.01
        }"#;

        assert!(lexer_dfa_accepts(&reciprocal_power_of_ten_dfa(2), b"10.99"));
        assert!(accepts_sequence(schema, &[b"10.99"]));
        assert!(accepts_sequence(schema, &[b"10.0"]));
        assert!(accepts_sequence(schema, &[b"10.000"]));
        assert!(accepts_sequence(schema, &[b"10"]));
        assert!(!accepts_sequence(schema, &[b"10.001"]));
        assert!(!accepts_sequence(schema, &[b"-10.0"]));
    }

    #[test]
    fn test_power_of_ten_multiple_of_rejects_extra_precision_without_bounds() {
        let schema = r#"{
            "type": "number",
            "multipleOf": 0.01
        }"#;

        assert!(lexer_dfa_accepts(&reciprocal_power_of_ten_dfa(2), b"10.0"));
        assert!(accepts_sequence(schema, &[b"10"]));
        assert!(accepts_sequence(schema, &[b"10.0"]));
        assert!(accepts_sequence(schema, &[b"10.000"]));
        assert!(!accepts_sequence(schema, &[b"10.001"]));
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

    fn regex_dfa_accepts(pattern: &str, text: &[u8]) -> bool {
        let dfa = build_regex(&[parse_regex(pattern, true)]).dfa;
        let mut state = 0u32;
        for &byte in text {
            let Some(next) = dfa.step(state, byte) else {
                return false;
            };
            state = next;
        }
        dfa.finalizers(state).contains(0)
    }

    fn lexer_dfa_accepts(dfa: &LexerDfa, text: &[u8]) -> bool {
        let mut state = 0u32;
        for &byte in text {
            let Some(next) = dfa.step(state, byte) else {
                return false;
            };
            state = next;
        }
        !dfa.finalizers(state).is_empty()
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

    #[test]
    fn test_anyof_enum_intersection_constrains_branches() {
        // When merging base enum [A,B,C] with branch enum [A], result should be [A] not [A,B,C].
        let schema = r##"{
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["A", "B", "C"]},
                "value": {"type": "integer"}
            },
            "anyOf": [
                {"properties": {"kind": {"enum": ["A"]}, "data": {"type": "string"}}},
                {"properties": {"kind": {"enum": ["B"]}, "data": {"type": "number"}}}
            ],
            "required": ["kind"]
        }"##;
        // kind=A with string data should be accepted
        assert!(accepts_sequence(schema, &[br#"{"kind": "A", "data": "x"}"#]));
        // kind=C should NOT be accepted (not in any anyOf branch)
        assert!(!accepts_sequence(schema, &[br#"{"kind": "C"}"#]));
    }

    #[test]
    fn test_properties_plus_pattern_properties_allows_pattern_keys() {
        // When properties + patternProperties + additionalProperties: false,
        // keys matching the pattern should be accepted.
        let schema = r#"{
            "type": "object",
            "properties": {
                "default": {"type": "string"}
            },
            "patternProperties": {
                "^[1-5][0-9]{2}$": {"type": "string"}
            },
            "additionalProperties": false
        }"#;
        // Explicit property should work
        assert!(accepts_sequence(schema, &[br#"{"default": "err.html"}"#]));
        // Pattern-matching key should work
        assert!(accepts_sequence(schema, &[br#"{"default": "err.html", "500": "server_error.html"}"#]));
        // Non-matching key should be rejected
        assert!(!accepts_sequence(schema, &[br#"{"default": "err.html", "abc": "x"}"#]));
    }

    #[test]
    #[ignore = "any-order object key acceptance not yet wired in"]
    fn test_literal_properties_accept_any_order() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a", "b"]
        }"#;

        assert!(accepts_sequence(schema, &[br#"{"a": "1", "b": "2"}"#]));
        assert!(accepts_sequence(schema, &[br#"{"b": "2", "a": "1"}"#]));
    }

    #[test]
    #[ignore = "any-order object key acceptance not yet wired in"]
    fn test_additional_properties_can_precede_declared_keys() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "known": {"type": "string"}
            }
        }"#;

        assert!(accepts_sequence(schema, &[br#"{"known": "x", "extra": "y"}"#]));
        assert!(accepts_sequence(schema, &[br#"{"extra": "y", "known": "x"}"#]));
    }

    #[test]
    #[ignore = "any-order object key acceptance not yet wired in"]
    fn test_allof_properties_accept_declared_keys_after_unknown_keys() {
        let schema = r##"{
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "definitions": {
                "Thing": {
                    "type": "object",
                    "properties": {
                        "additionalType": {"type": "string"},
                        "name": {"type": "string"}
                    }
                }
            },
            "allOf": [
                {"$ref": "#/definitions/Thing"}
            ]
        }"##;

        assert!(accepts_sequence(
            schema,
            &[br#"{"@type": "OfferItemCondition", "additionalType": "x", "name": "n"}"#],
        ));
    }

    #[test]
    fn test_literal_property_inherits_all_matching_pattern_properties() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "s_id": {"type": "string"}
            },
            "required": ["s_id"],
            "patternProperties": {
                "^s_": {"minLength": 3},
                "_id$": {"pattern": "^[A-Z]+$"}
            },
            "additionalProperties": false
        }"#;

        assert!(accepts_sequence(schema, &[br#"{"s_id": "ABC"}"#]));
        assert!(!accepts_sequence(schema, &[br#"{"s_id": "AB"}"#]));
        assert!(!accepts_sequence(schema, &[br#"{"s_id": "AbC"}"#]));
    }

    #[test]
    fn test_unanchored_pattern_with_max_length_rejects_non_matching() {
        // Regression: unanchored pattern [0-9]{10,10} with maxLength=10
        // must reject non-digit strings like "egg". Previously the string
        // tails were each allowed maxLength chars, defeating the constraint.
        let schema = r#"{
            "type": "string",
            "pattern": "[0-9]{10,10}",
            "minLength": 10,
            "maxLength": 10
        }"#;
        // 10 digits should be accepted
        assert!(accepts_sequence(schema, &[br#""1234567890""#]));
        // Non-digit string should be rejected
        assert!(!accepts_sequence(schema, &[br#""egg""#]));
        assert!(!accepts_sequence(schema, &[br#""abcdefghij""#]));
    }

    #[test]
    fn test_unanchored_pattern_with_max_length_rejects_overlong_match() {
        // Regression: with an unanchored pattern and maxLength, the importer
        // must keep the combined prefix+suffix slack within the remaining
        // length budget, not apply that budget independently to both sides.
        let schema = r#"{
            "type": "string",
            "pattern": "[0-9a-fA-F]+",
            "minLength": 12,
            "maxLength": 12
        }"#;
        let search_regex = string_value_body_regex(&json_search_pattern("[0-9a-fA-F]+"));
        let length_regex = json_wrapped_string_length_regex(12, 12);
        let test_input = string_value_literal_body_bytes("0123456789ab");
        assert!(regex_dfa_accepts(&search_regex, &test_input));
        assert!(regex_dfa_accepts(&length_regex, &test_input));
        assert!(accepts_sequence(schema, &[br#""0123456789ab""#]));
        assert!(!accepts_sequence(schema, &[br#""0123456789abc""#]));
        assert!(!accepts_sequence(schema, &[br#""0123456789abz""#]));
    }

    #[test]
    fn test_anchored_pattern_with_max_length_control() {
        // Anchored pattern: already works correctly (no tails generated).
        let schema = r#"{
            "type": "string",
            "pattern": "^[0-9]{10,10}$",
            "minLength": 10,
            "maxLength": 10
        }"#;
        assert!(accepts_sequence(schema, &[br#""1234567890""#]));
        assert!(!accepts_sequence(schema, &[br#""egg""#]));
        assert!(!accepts_sequence(schema, &[br#""abcdefghij""#]));
    }

    #[test]
    fn test_pattern_min_char_count_basic() {
        assert_eq!(pattern_min_char_count("[0-9]{10,10}"), Some(10));
        assert_eq!(pattern_min_char_count("abc"), Some(3));
        assert_eq!(pattern_min_char_count("a|bc"), Some(1));
        assert_eq!(pattern_min_char_count(".{5,10}"), Some(5));
        assert_eq!(pattern_min_char_count("^[0-9]+$"), Some(1));
        assert_eq!(pattern_min_char_count("\\d{3}-\\d{4}"), Some(8));
        assert_eq!(pattern_min_char_count("(?:a|bb)+"), Some(1));
        assert_eq!(pattern_min_char_count(""), Some(0));
        assert_eq!(pattern_min_char_count("a*"), Some(0));
    }

    #[test]
    fn test_pattern_min_char_count_returns_none_for_unsupported() {
        // Backreference — min length depends on captured group content
        assert_eq!(pattern_min_char_count("(a)\\1"), None);
        // Unicode property escape
        assert_eq!(pattern_min_char_count("\\p{L}+"), None);
        // Named backreference
        assert_eq!(pattern_min_char_count("(?P<n>a)\\k<n>"), None);
    }

    /// Regression test: the full string should also be accepted.
    #[test]
    fn test_structural_nested_object_array_full_string_accepted() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "maxLength": 10},
                            "b": {"type": "string", "maxLength": 10}
                        },
                        "required": ["a", "b"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": false
        }"#;
        assert!(accepts_sequence(schema, &[b"{\"items\": [{\"a\": \"x\", \"b\": \"y\"}]}"]));
    }

    /// Regression test: mask and commit must agree for every token at every step.
    #[test]
    fn test_structural_import_mask_commit_consistency() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "maxLength": 10},
                            "b": {"type": "string", "maxLength": 10}
                        },
                        "required": ["a", "b"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": false
        }"#;

        // The sequence that should be parseable with structural tokens
        let entries: Vec<(u32, Vec<u8>)> = vec![
            (0, b"{".to_vec()),
            (1, b"{\"".to_vec()),
            (2, b"items\"".to_vec()),
            (3, b": ".to_vec()),
            (4, b"[".to_vec()),
            (5, b"]".to_vec()),
            (6, b"}".to_vec()),
            (7, b"\"".to_vec()),
            (8, b", \"".to_vec()),
            (9, b"a\"".to_vec()),
            (10, b"b\"".to_vec()),
            (11, b"x".to_vec()),
        ];
        let vocab = Vocab::new(entries.clone(), None);
        let c = crate::Constraint::from_json_schema(schema, &vocab).unwrap();
        let state = c.start();
        let mask = state.mask();

        // For every token the mask says is valid, commit_token must succeed.
        for (id, bytes) in &entries {
            let wi = *id as usize / 32;
            let bi = *id as usize % 32;
            let allowed = wi < mask.len() && (mask[wi] >> bi) & 1 != 0;
            if allowed {
                let bytes_str = String::from_utf8_lossy(bytes);
                let mut state_clone = state.clone();
                let commit_result = state_clone.commit_token(*id);
                assert!(
                    commit_result.is_ok(),
                    "mask allows token {} ({:?}) but commit_token failed: {:?}",
                    id, bytes_str, commit_result.err()
                );
            }
        }
    }

    /// Regression test for o76439: cross-token terminal matching when `,` ends
    /// one token and ` {"` starts the next. The structural import creates
    /// terminal `, ` (comma-space) which spans this token boundary.
    #[test]
    fn test_structural_cross_token_comma_space_brace_quote() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "maxLength": 10},
                            "b": {"type": "string", "maxLength": 10}
                        },
                        "required": ["a", "b"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": false
        }"#;
        // Split so `,` ends one token and ` {"` starts the next (cross-token terminal matching).
        assert!(accepts_sequence(schema, &[
            b"{\"items\": [{\"a\": \"x\", \"b\": \"y\"},",
            b" {\"a\": \"z\", \"b\": \"w\"}]}",
        ]));
    }

    /// Same test as above but with the actual o76439 schema shape (not all
    /// outer properties required).
    #[test]
    fn test_o76439_cross_token_array_separator() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "maxLength": 512},
                            "note": {"type": "string", "maxLength": 512}
                        },
                        "required": ["id", "note"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["severity"],
            "additionalProperties": false
        }"#;
        // The comma at the end of the first token spans across to the space at the start of the second.
        assert!(accepts_sequence(schema, &[
            b"{\"severity\": \"low\", \"items\": [{\"id\": \"a\", \"note\": \"b\"},",
            b" {\"id\": \"c\", \"note\": \"d\"}]}",
        ]));
    }

}
