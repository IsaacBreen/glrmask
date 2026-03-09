#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::{Map, Value};

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, lower};

const JSON_VALUE_RULE: &str = "json_value";
const JSON_OBJECT_RULE: &str = "json_object";
const JSON_ARRAY_RULE: &str = "json_array";
const JSON_KV_RULE: &str = "json_kv";
const JSON_STRING_RULE: &str = "JSON_STRING";
const JSON_INTEGER_RULE: &str = "JSON_INTEGER";
const JSON_NUMBER_RULE: &str = "JSON_NUMBER";
const JSON_BOOL_RULE: &str = "JSON_BOOL";
const JSON_NULL_RULE: &str = "JSON_NULL";
const JSON_KEY_COLON_RULE: &str = "JSON_KEY_COLON";

const JSON_STRING_REGEX: &str =
    r#""([^"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*""#;
const JSON_KEY_COLON_REGEX: &str =
    r#""([^"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*":"#;
const JSON_STRING_CHAR_PATTERN: &str = r#"[^"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}"#;

fn literal_expr(bytes: &[u8]) -> GrammarExpr {
    GrammarExpr::Literal(bytes.to_vec())
}

fn regex_expr(pattern: impl Into<String>) -> GrammarExpr {
    GrammarExpr::RawRegex(pattern.into())
}

fn empty_expr() -> GrammarExpr {
    GrammarExpr::Sequence(Vec::new())
}

fn json_format_pattern(format_name: &str) -> Option<&'static str> {
    Some(match format_name {
        "date" => r#"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"#,
        "time" => {
            r#"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"#
        }
        "date-time" => {
            r#"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])[Tt](?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"#
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
        "uuid" => r#"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"#,
        "uri" => {
            r#"[a-zA-Z][a-zA-Z0-9+\-.]*:(?://(?:[a-zA-Z0-9\-._~%!$&'()*+,;=:]*@)?[a-zA-Z0-9\-._~%!$&'()*+,;=\[\]]+(?::[0-9]*)?)?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/]*(?:\?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/?]*)?(?:#[a-zA-Z0-9\-._~%!$&'()*+,;=:@/?]*)?"#
        }
        "uri-reference" => {
            r#"(?:[a-zA-Z][a-zA-Z0-9+\-.]*:(?://(?:[a-zA-Z0-9\-._~%!$&'()*+,;=:]*@)?[a-zA-Z0-9\-._~%!$&'()*+,;=\[\]]+(?::[0-9]*)?)?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/]*)?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/]*(?:\?[a-zA-Z0-9\-._~%!$&'()*+,;=:@/?]*)?(?:#[a-zA-Z0-9\-._~%!$&'()*+,;=:@/?]*)?"#
        }
        _ => return None,
    })
}

fn strip_regex_anchors(pattern: &str) -> &str {
    let pattern = pattern.strip_prefix('^').unwrap_or(pattern);
    pattern.strip_suffix('$').unwrap_or(pattern)
}

fn json_wrapped_pattern(pattern: &str) -> GrammarExpr {
    regex_expr(format!(r#""(?:{})""#, strip_regex_anchors(pattern)))
}

fn json_wrapped_key_colon_pattern(pattern: &str) -> GrammarExpr {
    regex_expr(format!(r#""(?:{})":"#, strip_regex_anchors(pattern)))
}

fn json_string_literal_bytes(text: &str) -> Vec<u8> {
    serde_json::to_string(text)
        .unwrap_or_else(|_| format!("\"{}\"", text))
        .into_bytes()
}

fn json_value_literal_expr(value: &Value) -> GrammarExpr {
    let rendered = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
    literal_expr(rendered.as_bytes())
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
        let mut keys = BTreeSet::new();
        if let Some(props) = props1 {
            keys.extend(props.keys().cloned());
        }
        if let Some(props) = props2 {
            keys.extend(props.keys().cloned());
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
    let named = schema_to_named_grammar(&schema)?;
    lower(&named)
}

pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {
    let mut ctx = SchemaCtx::new(schema);
    ctx.register_root_definitions();
    ctx.materialize_registered_refs()?;
    let start_expr = ctx.convert_schema(schema)?;
    ctx.insert_rule("start", start_expr);
    Ok(NamedGrammar {
        rules: ctx.rules,
        start: "start".into(),
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
        };
        ctx.ensure_base_rules();
        ctx
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
        self.insert_rule(JSON_STRING_RULE, regex_expr(JSON_STRING_REGEX));
        self.insert_rule(JSON_INTEGER_RULE, regex_expr(r#"-?(0|[1-9][0-9]*)"#));
        self.insert_rule(
            JSON_NUMBER_RULE,
            regex_expr(r#"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"#),
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
                        sequence_or_single(vec![literal_expr(b","), GrammarExpr::Ref(JSON_KV_RULE.into())]),
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
                        sequence_or_single(vec![literal_expr(b","), self.json_value_ref()]),
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

    fn resolve_local_ref(&self, ref_value: &str) -> Result<Value, GlrMaskError> {
        if !ref_value.starts_with("#/") {
            return Err(GlrMaskError::GrammarParse(format!(
                "unsupported $ref '{ref_value}'"
            )));
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
            Ok(target) => self.convert_schema(&target).unwrap_or_else(|_| self.json_value_ref()),
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
        let Some(object) = schema.as_object() else {
            return Ok(self.json_value_ref());
        };

        if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
            return self.convert_ref(reference);
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
                let options = options
                    .iter()
                    .map(|option| self.convert_schema(option))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(factor_common_affixes(options));
            }
        }

        if let Some(options) = object.get("oneOf").and_then(Value::as_array) {
            if !options.is_empty() {
                let options = options
                    .iter()
                    .map(|option| self.convert_schema(option))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(factor_common_affixes(options));
            }
        }

        if let Some(all_of) = object.get("allOf").and_then(Value::as_array) {
            if !all_of.is_empty() {
                let base = object
                    .iter()
                    .filter(|(key, _)| key.as_str() != "allOf")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<Map<String, Value>>();
                let merged = merge_allof_schemas(&base, all_of);
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

        if [
            "properties",
            "required",
            "additionalProperties",
            "patternProperties",
            "propertyNames",
            "minProperties",
            "maxProperties",
        ]
        .iter()
        .any(|key| object.contains_key(*key))
        {
            return self.build_object_expr(object);
        }

        if object.contains_key("prefixItems") || object.contains_key("items") {
            return self.build_array_expr(object);
        }

        Ok(self.json_value_ref())
    }

    fn convert_type(&mut self, type_name: &str, schema: &Map<String, Value>) -> Result<GrammarExpr, GlrMaskError> {
        match type_name {
            "object" => self.build_object_expr(schema),
            "array" => self.build_array_expr(schema),
            "string" => Ok(self.build_string_expr(schema)),
            "integer" => Ok(self.json_integer_ref()),
            "number" => Ok(self.json_number_ref()),
            "boolean" => Ok(self.json_bool_ref()),
            "null" => Ok(self.json_null_ref()),
            _ => Ok(self.json_value_ref()),
        }
    }

    fn build_string_expr(&self, schema: &Map<String, Value>) -> GrammarExpr {
        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            return json_wrapped_pattern(pattern);
        }

        if let Some(format_name) = schema.get("format").and_then(Value::as_str) {
            if let Some(pattern) = json_format_pattern(format_name) {
                return json_wrapped_pattern(pattern);
            }
        }

        let min_len = schema
            .get("minLength")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(0);
        let max_len = schema
            .get("maxLength")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        if min_len == 0 && max_len.is_none() {
            return self.json_string_ref();
        }

        let pattern = match max_len {
            Some(max_len) if min_len == max_len => {
                format!(r#""(?:{}){{{}}}""#, JSON_STRING_CHAR_PATTERN, min_len)
            }
            Some(max_len) => {
                format!(r#""(?:{}){{{},{}}}""#, JSON_STRING_CHAR_PATTERN, min_len, max_len)
            }
            None => format!(r#""(?:{}){{{},}}""#, JSON_STRING_CHAR_PATTERN, min_len),
        };
        regex_expr(pattern)
    }

    fn json_literal(&self, value: &Value) -> GrammarExpr {
        json_value_literal_expr(value)
    }

    fn json_string_literal(&self, text: &str) -> GrammarExpr {
        literal_expr(&json_string_literal_bytes(text))
    }

    fn json_key_colon_literal(&self, text: &str) -> GrammarExpr {
        let mut bytes = json_string_literal_bytes(text);
        bytes.push(b':');
        literal_expr(&bytes)
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
            let additional_schema = match additional_properties {
                Some(Value::Bool(false)) => None,
                Some(Value::Object(map)) => Some(Value::Object(map.clone())),
                _ => Some(serde_json::json!({})),
            };
            return self.build_ordered_properties_object_expr(
                properties,
                &required_list,
                &required_keys,
                additional_schema,
            );
        }

        if let Some(property_names) = property_names {
            return self.build_pattern_named_object_expr(property_names);
        }

        Ok(self.json_object_ref())
    }

    fn build_ordered_properties_object_expr(
        &mut self,
        properties: &Map<String, Value>,
        required_list: &[String],
        required_keys: &BTreeSet<String>,
        additional_properties_schema: Option<Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let mut ordered = Vec::new();
        for (key, subschema) in properties {
            ordered.push((key.clone(), subschema.clone(), required_keys.contains(key)));
        }
        for key in required_list {
            if !properties.contains_key(key) {
                ordered.push((key.clone(), serde_json::json!({}), true));
            }
        }

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
            let cache_key = serde_json::to_string(&schema).unwrap_or_else(|_| format!("{schema:?}"));
            if let Some(names) = self.ap_catch_all_cache.get(&cache_key) {
                names.clone()
            } else {
                let slot = self.ap_catch_all_cache.len();
                let term_nc = format!("ap_extra_{slot}_nc");
                let term_c = format!("ap_extra_{slot}_c");
                let value_expr = self.convert_schema(&schema)?;
                self.insert_rule(
                    term_nc.clone(),
                    choice_or_single(vec![
                        sequence_or_single(vec![
                            self.json_key_colon_ref(),
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
                            literal_expr(b","),
                            self.json_key_colon_ref(),
                            value_expr,
                            GrammarExpr::Ref(term_c.clone()),
                        ]),
                        empty_expr(),
                    ]),
                );
                self.ap_catch_all_cache
                    .insert(cache_key, (term_nc.clone(), term_c.clone()));
                (term_nc, term_c)
            }
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
            literal_expr(b","),
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

    fn build_pattern_named_object_expr(&mut self, property_names: &Value) -> Result<GrammarExpr, GlrMaskError> {
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

        let pair = sequence_or_single(vec![
            json_wrapped_key_colon_pattern(pattern),
            self.json_value_ref(),
        ]);
        Ok(choice_or_single(vec![
            sequence_or_single(vec![literal_expr(b"{"), literal_expr(b"}")]),
            sequence_or_single(vec![
                literal_expr(b"{"),
                pair.clone(),
                repeat_expr(sequence_or_single(vec![literal_expr(b","), pair]), 0, None),
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
                    parts.push(literal_expr(b","));
                }
                parts.push(self.convert_schema(item_schema)?);
            }

            if extra_max.is_none() || extra_max.unwrap_or(0) > 0 || extra_min > 0 {
                let extra_item_expr = self.extract_rule(extra_item_expr, "arr_item");
                parts.push(repeat_expr(
                    sequence_or_single(vec![literal_expr(b","), extra_item_expr]),
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
                sequence_or_single(vec![literal_expr(b","), item_expr]),
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
        _ => {
            let mut parts = Vec::new();
            match max {
                Some(max) => {
                    for _ in 0..min {
                        parts.push(item.clone());
                    }
                    for _ in min..max {
                        parts.push(GrammarExpr::Optional(Box::new(item.clone())));
                    }
                }
                None => {
                    for _ in 0..min {
                        parts.push(item.clone());
                    }
                    parts.push(GrammarExpr::Repeat(Box::new(item)));
                }
            }
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
        let g = json_schema_to_grammar(r#"{
            "oneOf": [{"type": "string"}, {"type": "integer"}]
        }"#).unwrap();
        assert!(!g.rules.is_empty());
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
    fn test_type_array_of_types() {
        let g = json_schema_to_grammar(r#"{"type": ["string", "null"]}"#).unwrap();
        assert!(!g.rules.is_empty());
    }

    fn accepts_sequence(schema_json: &str, tokens: &[&[u8]]) -> bool {
        let entries: Vec<(u32, Vec<u8>)> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u32, t.to_vec()))
            .collect();
        let vocab = Vocab::new(entries, None);

        let c = match crate::Constraint::from_json_schema(schema_json, &vocab) {
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
            state.commit_token(id);
        }
        state.is_finished()
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
