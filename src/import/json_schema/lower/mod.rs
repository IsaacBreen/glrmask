//! Schema-to-grammar lowering.
//!
//! The lowerer interprets loaded JSON Schema assertions as languages of encoded
//! JSON texts and emits the crate's grammar IR.  It is the only JSON Schema
//! importer phase allowed to construct `GrammarExpr` values.

pub(crate) mod array;
pub(crate) mod number;
pub(crate) mod object;
pub(crate) mod string;

use std::collections::{BTreeMap, BTreeSet};

use regex::escape as regex_escape;
use serde_json::Value;

use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule};

use super::schema::{
    AdditionalProperties, Schema, SchemaAssertions, SchemaDocument, SchemaKind, SchemaType,
};
use super::options::JsonSchemaConfig;
use super::diagnostics::{ImportResult, SchemaImportError};
use self::string::string_value_satisfies_schema;

pub(crate) const JSON_VALUE_RULE: &str = "json_value";
pub(crate) const JSON_OBJECT_RULE: &str = "json_object";
pub(crate) const JSON_ARRAY_RULE: &str = "json_array";
pub(crate) const JSON_STRING_RULE: &str = "JSON_STRING";
pub(crate) const JSON_STRING_CHAR_RULE: &str = "JSON_STRING_CHAR";
pub(crate) const JSON_ITEM_SEPARATOR_RULE: &str = "JSON_ITEM_SEPARATOR";
pub(crate) const JSON_KEY_SEPARATOR_RULE: &str = "JSON_KEY_SEPARATOR";
pub(crate) const JSON_INTEGER_RULE: &str = "JSON_INTEGER";
pub(crate) const JSON_NUMBER_RULE: &str = "JSON_NUMBER";
pub(crate) const JSON_BOOL_RULE: &str = "JSON_BOOL";
pub(crate) const JSON_NULL_RULE: &str = "JSON_NULL";
pub(crate) const JSON_ADDITIONAL_KEY_COLON_SHARED_RULE: &str = "JSON_ADDITIONAL_KEY_COLON_SHARED";
pub(crate) const JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE: &str =
    "JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED";
pub(crate) const JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE: &str =
    "json_additional_excluded_key_colon_shared";
pub(crate) const MAX_SHARED_ADDITIONAL_EXCLUSION_KEYS: usize = 256;
const STRING_ENUM_REGEX_MIN_VALUES: usize = 64;
const STRING_ENUM_REGEX_MIN_ENCODED_BYTES: usize = 1024;

pub(crate) fn lower_document(
    document: &SchemaDocument,
    config: JsonSchemaConfig,
) -> ImportResult<NamedGrammar> {
    let lowerer = Lowerer::new(document, config);
    lowerer.finish()
}

pub(crate) struct Lowerer<'a> {
    pub(crate) document: &'a SchemaDocument,
    pub(crate) config: JsonSchemaConfig,
    pub(crate) rules: Vec<NamedRule>,
    pub(crate) shared_string_exact_rules: BTreeMap<usize, String>,
    pub(crate) shared_string_upto_rules: BTreeMap<usize, String>,
    pub(crate) shared_string_upto_close_rules: BTreeMap<usize, String>,
    pub(crate) shared_ap_literal_keys: BTreeSet<String>,
    pub(crate) shared_ap_patterns: Vec<String>,
    pub(crate) shared_ap_base_rule: Option<String>,
    pub(crate) shared_ap_excluded_rule: Option<String>,
    pub(crate) shared_ap_pattern_rules: BTreeMap<String, String>,
    pub(crate) shared_pattern_overlap_keys: BTreeMap<String, Vec<String>>,
    pub(crate) shared_pattern_overlap_literal_rules: BTreeMap<String, String>,
    pub(crate) shared_pattern_appearance_rules: BTreeMap<(String, Vec<String>), String>,
    definition_rules: BTreeMap<String, String>,
    definition_by_pointer: BTreeMap<String, &'a Schema>,
    used_rule_names: BTreeSet<String>,
    next_rule_id: usize,
}

impl<'a> Lowerer<'a> {
    fn new(document: &'a SchemaDocument, config: JsonSchemaConfig) -> Self {
        let (shared_ap_literal_keys, shared_ap_patterns) = collect_shared_ap_exclusion_plan(document);
        let mut definition_by_pointer = BTreeMap::new();
        for definition in &document.definitions {
            definition_by_pointer.insert(definition.pointer.clone(), &definition.schema);
        }
        for target in &document.ref_targets {
            definition_by_pointer.insert(target.pointer.clone(), &target.schema);
        }
        definition_by_pointer.insert("#".to_string(), &document.root);

        let mut lowerer = Self {
            document,
            config,
            rules: Vec::new(),
            shared_string_exact_rules: BTreeMap::new(),
            shared_string_upto_rules: BTreeMap::new(),
            shared_string_upto_close_rules: BTreeMap::new(),
            shared_ap_literal_keys,
            shared_ap_patterns,
            shared_ap_base_rule: None,
            shared_ap_excluded_rule: None,
            shared_ap_pattern_rules: BTreeMap::new(),
            shared_pattern_overlap_keys: BTreeMap::new(),
            shared_pattern_overlap_literal_rules: BTreeMap::new(),
            shared_pattern_appearance_rules: BTreeMap::new(),
            definition_rules: BTreeMap::new(),
            definition_by_pointer,
            used_rule_names: BTreeSet::new(),
            next_rule_id: 0,
        };
        lowerer.install_json_builtins();
        lowerer
    }

    fn finish(mut self) -> ImportResult<NamedGrammar> {
        let start_expr = self.lower_schema(&self.document.root)?;
        self.add_nonterminal_rule("start", start_expr);
        Ok(NamedGrammar { rules: self.rules, start: "start".to_string(), ignore: None })
    }

    fn install_json_builtins(&mut self) {
        let string_char = self.json_string_char_regex();
        self.add_terminal_rule(
            JSON_STRING_CHAR_RULE,
            GrammarExpr::RawRegex(string_char.clone()),
        );
        self.add_terminal_rule(
            JSON_STRING_RULE,
            GrammarExpr::RawRegex(format!(r#""(?:{string_char})*""#)),
        );
        self.add_terminal_rule(
            JSON_ITEM_SEPARATOR_RULE,
            GrammarExpr::RawRegex(self.separator_regex(",")),
        );
        self.add_terminal_rule(
            JSON_KEY_SEPARATOR_RULE,
            GrammarExpr::RawRegex(self.separator_regex(":")),
        );
        self.add_terminal_rule(
            JSON_INTEGER_RULE,
            GrammarExpr::RawRegex(r#"-?(0|[1-9][0-9]*)"#.to_string()),
        );
        self.add_terminal_rule(
            JSON_NUMBER_RULE,
            GrammarExpr::RawRegex(r#"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"#.to_string()),
        );
        self.add_terminal_rule(
            JSON_BOOL_RULE,
            choice(vec![lit("true"), lit("false")]),
        );
        self.add_terminal_rule(JSON_NULL_RULE, lit("null"));

        let array_item_tail = seq(vec![r(JSON_ITEM_SEPARATOR_RULE), r(JSON_VALUE_RULE)]);
        self.add_nonterminal_rule(
            JSON_ARRAY_RULE,
            seq(vec![
                lit("["),
                GrammarExpr::Optional(Box::new(seq(vec![
                    r(JSON_VALUE_RULE),
                    GrammarExpr::Repeat(Box::new(array_item_tail)),
                ]))),
                lit("]"),
            ]),
        );

        let object_entry = seq(vec![r(JSON_STRING_RULE), r(JSON_KEY_SEPARATOR_RULE), r(JSON_VALUE_RULE)]);
        let object_tail = seq(vec![r(JSON_ITEM_SEPARATOR_RULE), object_entry.clone()]);
        self.add_nonterminal_rule(
            JSON_OBJECT_RULE,
            seq(vec![
                lit("{"),
                GrammarExpr::Optional(Box::new(seq(vec![
                    object_entry,
                    GrammarExpr::Repeat(Box::new(object_tail)),
                ]))),
                lit("}"),
            ]),
        );

        self.add_nonterminal_rule(
            JSON_VALUE_RULE,
            choice(vec![
                r(JSON_NULL_RULE),
                r(JSON_BOOL_RULE),
                r(JSON_NUMBER_RULE),
                r(JSON_STRING_RULE),
                r(JSON_ARRAY_RULE),
                r(JSON_OBJECT_RULE),
            ]),
        );
    }

    pub(crate) fn item_separator_expr(&self) -> GrammarExpr {
        r(JSON_ITEM_SEPARATOR_RULE)
    }

    pub(crate) fn key_separator_expr(&self) -> GrammarExpr {
        r(JSON_KEY_SEPARATOR_RULE)
    }

    fn separator_regex(&self, separator: &str) -> String {
        match separator {
            "," | ":" => format!("(?:{separator} )"),
            _ => format!("(?:{separator})"),
        }
    }

    fn json_string_char_regex(&self) -> String {
        r#"[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt]"#.to_string()
    }

    pub(crate) fn lower_schema(&mut self, schema: &Schema) -> ImportResult<GrammarExpr> {
        match &schema.kind {
            SchemaKind::Any => Ok(r(JSON_VALUE_RULE)),
            SchemaKind::Never => Ok(never()),
            SchemaKind::Ref(pointer) => self.lower_ref(pointer),
            SchemaKind::Assertions(assertions) => self.lower_assertions(schema, assertions),
        }
    }

    pub(crate) fn lower_ref(&mut self, pointer: &str) -> ImportResult<GrammarExpr> {
        let normalized = normalize_local_ref(pointer)?;
        if normalized == "#" {
            return Ok(r("start"));
        }
        if let Some(rule_name) = self.definition_rules.get(&normalized) {
            return Ok(r(rule_name));
        }

        let target = *self.definition_by_pointer.get(&normalized).ok_or_else(|| {
            SchemaImportError::new(format!("unsupported or unresolved local $ref {pointer:?}"))
        })?;

        let rule_name = self.fresh_rule_name("schema_ref");
        self.definition_rules.insert(normalized, rule_name.clone());
        let expr = self.lower_schema(target)?;
        self.add_nonterminal_rule(&rule_name, expr);
        Ok(r(&rule_name))
    }

    pub(crate) fn resolve_ref_target(&self, pointer: &str) -> ImportResult<&'a Schema> {
        let normalized = normalize_local_ref(pointer)?;
        self.definition_by_pointer
            .get(&normalized)
            .copied()
            .ok_or_else(|| SchemaImportError::new(format!("unsupported or unresolved local $ref {pointer:?}")))
    }

    fn lower_assertions(
        &mut self,
        schema: &Schema,
        assertions: &SchemaAssertions,
    ) -> ImportResult<GrammarExpr> {
        if !assertions.all_of.is_empty() {
            return self.lower_all_of(assertions);
        }
        if !assertions.any_of.is_empty() {
            return self.lower_any_of(schema, assertions);
        }
        if !assertions.one_of.is_empty() {
            return self.lower_one_of(assertions);
        }
        if assertions.not.is_some() {
            return Err(SchemaImportError::at(
                &schema.location,
                "not is only supported for mutually exclusive object-property anyOf branches",
            ));
        }

        if let Some(value) = &assertions.const_value {
            return Ok(self.lower_json_literal(value));
        }
        if let Some(encoded_literals) = large_string_enum_regex_literals(assertions)? {
            return Ok(GrammarExpr::RawRegex(string_enum_regex(&encoded_literals)));
        }
        if let Some(values) = &assertions.enum_values {
            let values = if let Some(string_schema) = &assertions.string {
                values
                    .iter()
                    .filter(|value| string_value_satisfies_schema(value, string_schema).unwrap_or(false))
                    .collect::<Vec<_>>()
            } else {
                values.iter().collect::<Vec<_>>()
            };
            if let Some(expr) = factored_small_string_enum_expr(&values) {
                return Ok(expr);
            }
            return Ok(choice(values.into_iter().map(|value| self.lower_json_literal(value)).collect()));
        }

        if assertions.types.is_none() {
            let inferred = self.inferred_constrained_types(assertions);
            if inferred.len() == 1 {
                return self.lower_untyped_single_family_assertions(inferred[0], assertions);
            }
        }

        let selected_types = self.selected_types(schema, assertions)?;
        if selected_types.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }

        let alternatives = selected_types
            .into_iter()
            .map(|schema_type| self.lower_for_type(schema_type, assertions))
            .collect::<ImportResult<Vec<_>>>()?;
        Ok(choice(alternatives))
    }

    fn selected_types(
        &self,
        schema: &Schema,
        assertions: &SchemaAssertions,
    ) -> ImportResult<Vec<SchemaType>> {
        if let Some(types) = &assertions.types {
            return Ok(types.clone());
        }

        let inferred = self.inferred_constrained_types(assertions);

        if inferred.len() > 1 {
            return Err(SchemaImportError::at(
                &schema.location,
                "untyped schemas with constraints for multiple primitive families are unsupported",
            ));
        }
        Ok(inferred)
    }

    fn lower_for_type(
        &mut self,
        schema_type: SchemaType,
        assertions: &SchemaAssertions,
    ) -> ImportResult<GrammarExpr> {
        match schema_type {
            SchemaType::Null => Ok(r(JSON_NULL_RULE)),
            SchemaType::Boolean => Ok(r(JSON_BOOL_RULE)),
            SchemaType::Object => {
                let object = assertions.object.clone().unwrap_or_default();
                self.lower_object(&object)
            }
            SchemaType::Array => {
                let array = assertions.array.clone().unwrap_or_default();
                self.lower_array(&array)
            }
            SchemaType::String => {
                let string = assertions.string.clone().unwrap_or_default();
                self.lower_string(&string)
            }
            SchemaType::Number => {
                let number = assertions.number.clone().unwrap_or_default();
                self.lower_number(&number)
            }
            SchemaType::Integer => {
                let mut number = assertions.number.clone().unwrap_or_default();
                number.integer = true;
                self.lower_number(&number)
            }
        }
    }

    fn inferred_constrained_types(&self, assertions: &SchemaAssertions) -> Vec<SchemaType> {
        let mut inferred = Vec::new();
        if assertions.object.is_some() {
            inferred.push(SchemaType::Object);
        }
        if assertions.array.is_some() {
            inferred.push(SchemaType::Array);
        }
        if assertions.string.is_some() {
            inferred.push(SchemaType::String);
        }
        if assertions.number.is_some() {
            inferred.push(SchemaType::Number);
        }
        inferred
    }

    fn lower_untyped_single_family_assertions(
        &mut self,
        constrained_type: SchemaType,
        assertions: &SchemaAssertions,
    ) -> ImportResult<GrammarExpr> {
        let mut alternatives = vec![self.lower_for_type(constrained_type, assertions)?];

        for fallback_type in [
            SchemaType::Object,
            SchemaType::Array,
            SchemaType::String,
            SchemaType::Number,
            SchemaType::Boolean,
            SchemaType::Null,
        ] {
            if fallback_type == constrained_type {
                continue;
            }
            alternatives.push(match fallback_type {
                SchemaType::Object => r(JSON_OBJECT_RULE),
                SchemaType::Array => r(JSON_ARRAY_RULE),
                SchemaType::String => r(JSON_STRING_RULE),
                SchemaType::Number | SchemaType::Integer => r(JSON_NUMBER_RULE),
                SchemaType::Boolean => r(JSON_BOOL_RULE),
                SchemaType::Null => r(JSON_NULL_RULE),
            });
        }

        Ok(choice(alternatives))
    }

    pub(crate) fn lower_json_literal(&mut self, value: &Value) -> GrammarExpr {
        match value {
            Value::String(text) => self.lower_string_literal(text),
            Value::Null => lit("null"),
            Value::Bool(true) => lit("true"),
            Value::Bool(false) => lit("false"),
            Value::Number(_) => lit_bytes(
                serde_json::to_string(value)
                    .unwrap_or_else(|_| "null".to_string())
                    .into_bytes(),
            ),
            Value::Array(items) => {
                if items.is_empty() {
                    return seq(vec![lit("["), lit("]")]);
                }

                let mut parts = Vec::with_capacity(items.len() * 2 + 1);
                parts.push(lit("["));
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        parts.push(r(JSON_ITEM_SEPARATOR_RULE));
                    }
                    parts.push(self.lower_json_literal(item));
                }
                parts.push(lit("]"));
                seq(parts)
            }
            Value::Object(map) => {
                if map.is_empty() {
                    return seq(vec![lit("{"), lit("}")]);
                }

                let mut parts = Vec::with_capacity(map.len() * 3 + 1);
                parts.push(lit("{"));
                for (index, (key, item)) in map.iter().enumerate() {
                    if index > 0 {
                        parts.push(r(JSON_ITEM_SEPARATOR_RULE));
                    }
                    parts.push(self.lower_literal_key_colon(key));
                    parts.push(self.lower_json_literal(item));
                }
                parts.push(lit("}"));
                seq(parts)
            }
        }
    }

    pub(crate) fn add_nonterminal_rule(&mut self, name: &str, expr: GrammarExpr) {
        self.used_rule_names.insert(name.to_string());
        self.rules.push(NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        });
    }

    pub(crate) fn add_terminal_rule(&mut self, name: &str, expr: GrammarExpr) {
        self.used_rule_names.insert(name.to_string());
        self.rules.push(NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: false,
        });
    }

    pub(crate) fn add_internal_terminal_rule(&mut self, name: &str, expr: GrammarExpr) {
        self.used_rule_names.insert(name.to_string());
        self.rules.push(NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: true,
        });
    }

    pub(crate) fn fresh_rule_name(&mut self, prefix: &str) -> String {
        loop {
            let candidate = format!("{prefix}_{}", self.next_rule_id);
            self.next_rule_id += 1;
            if self.used_rule_names.insert(candidate.clone()) {
                return candidate;
            }
        }
    }
}

fn large_string_enum_regex_literals(assertions: &SchemaAssertions) -> ImportResult<Option<Vec<String>>> {
    let Some(values) = &assertions.enum_values else {
        return Ok(None);
    };
    if assertions.const_value.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.number.is_some()
    {
        return Ok(None);
    }
    if let Some(types) = &assertions.types
        && (types.len() != 1 || types[0] != SchemaType::String)
    {
        return Ok(None);
    }
    if assertions.string.as_ref().is_some_and(|schema| schema.pattern.is_some()) {
        return Ok(None);
    }

    let mut encoded_literals = Vec::new();
    for value in values {
        let Value::String(text) = value else {
            return Ok(None);
        };
        if let Some(string_schema) = &assertions.string
            && !string_value_satisfies_schema(value, string_schema)?
        {
            continue;
        }
        encoded_literals.push(serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string()));
    }

    if encoded_literals.is_empty() {
        return Ok(None);
    }

    let encoded_bytes = encoded_literals.iter().map(|literal| literal.len()).sum::<usize>();
    if encoded_literals.len() < STRING_ENUM_REGEX_MIN_VALUES
        && encoded_bytes < STRING_ENUM_REGEX_MIN_ENCODED_BYTES
    {
        return Ok(None);
    }

    Ok(Some(encoded_literals))
}

fn string_enum_regex(encoded_literals: &[String]) -> String {
    format!(
        "(?:{})",
        encoded_literals
            .iter()
            .map(|literal| regex_escape(literal))
            .collect::<Vec<_>>()
            .join("|")
    )
}

fn factored_small_string_enum_expr(values: &[&Value]) -> Option<GrammarExpr> {
    if values.len() < 2 {
        return None;
    }

    let mut suffixes = Vec::with_capacity(values.len());
    for value in values {
        let Value::String(text) = value else {
            return None;
        };
        let encoded = serde_json::to_string(text).ok()?;
        let bytes = encoded.as_bytes();
        if bytes.first().copied() != Some(b'"') || bytes.len() < 2 {
            return None;
        }
        suffixes.push(lit_bytes(bytes[1..].to_vec()));
    }

    Some(seq(vec![lit_bytes(vec![b'"']), choice(suffixes)]))
}

fn collect_shared_ap_exclusion_plan(document: &SchemaDocument) -> (BTreeSet<String>, Vec<String>) {
    let mut literal_keys = BTreeSet::new();
    let mut patterns = BTreeSet::new();

    collect_shared_ap_exclusions_from_schema(&document.root, &mut literal_keys, &mut patterns);
    for definition in &document.definitions {
        collect_shared_ap_exclusions_from_schema(&definition.schema, &mut literal_keys, &mut patterns);
    }
    for target in &document.ref_targets {
        collect_shared_ap_exclusions_from_schema(&target.schema, &mut literal_keys, &mut patterns);
    }

    (literal_keys, patterns.into_iter().collect())
}

fn collect_shared_ap_exclusions_from_schema(
    schema: &Schema,
    literal_keys: &mut BTreeSet<String>,
    patterns: &mut BTreeSet<String>,
) {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return;
    };

    if let Some(object) = &assertions.object {
        let include_object_keys =
            !matches!(object.additional_properties, AdditionalProperties::Deny);
        if include_object_keys {
            for required_name in &object.required {
                literal_keys.insert(required_name.clone());
            }
        }
        for property in &object.properties {
            if include_object_keys {
                literal_keys.insert(property.name.clone());
            }
            collect_shared_ap_exclusions_from_schema(&property.schema, literal_keys, patterns);
        }
        for pattern_property in &object.pattern_properties {
            if include_object_keys {
                patterns.insert(pattern_property.pattern.clone());
            }
            collect_shared_ap_exclusions_from_schema(&pattern_property.schema, literal_keys, patterns);
        }
        if let super::schema::AdditionalProperties::Schema(schema) = &object.additional_properties {
            collect_shared_ap_exclusions_from_schema(schema, literal_keys, patterns);
        }
    }

    if let Some(array) = &assertions.array {
        collect_shared_ap_exclusions_from_schema(&array.items, literal_keys, patterns);
        for item in &array.prefix_items {
            collect_shared_ap_exclusions_from_schema(item, literal_keys, patterns);
        }
    }

    for branch in &assertions.any_of {
        collect_shared_ap_exclusions_from_schema(branch, literal_keys, patterns);
    }
    for branch in &assertions.one_of {
        collect_shared_ap_exclusions_from_schema(branch, literal_keys, patterns);
    }
    for branch in &assertions.all_of {
        collect_shared_ap_exclusions_from_schema(branch, literal_keys, patterns);
    }
}

pub(crate) fn normalize_local_ref(pointer: &str) -> ImportResult<String> {
    if pointer == "#" {
        return Ok("#".to_string());
    }
    if pointer.starts_with("#/") || is_local_fragment_alias(pointer) || is_absolute_self_ref_alias(pointer) {
        return Ok(pointer.to_string());
    }
    Err(SchemaImportError::new(format!(
        "only local JSON pointer $ref values are supported, got {pointer:?}"
    )))
}

fn is_local_fragment_alias(pointer: &str) -> bool {
    pointer.starts_with("#") && !pointer.starts_with("#/")
}

fn is_absolute_self_ref_alias(pointer: &str) -> bool {
    pointer.contains("://") && pointer.ends_with("#")
}

pub(crate) fn r(name: &str) -> GrammarExpr {
    GrammarExpr::Ref(name.to_string())
}

pub(crate) fn lit(text: &str) -> GrammarExpr {
    lit_bytes(text.as_bytes().to_vec())
}

pub(crate) fn lit_bytes(bytes: Vec<u8>) -> GrammarExpr {
    GrammarExpr::Literal(bytes)
}

pub(crate) fn seq(mut parts: Vec<GrammarExpr>) -> GrammarExpr {
    match parts.len() {
        0 => GrammarExpr::Epsilon,
        1 => parts.pop().unwrap(),
        _ => GrammarExpr::Sequence(parts),
    }
}

pub(crate) fn choice(mut alternatives: Vec<GrammarExpr>) -> GrammarExpr {
    if alternatives
        .iter()
        .any(|expr| matches!(expr, GrammarExpr::Ref(name) if name == JSON_NUMBER_RULE))
    {
        alternatives
            .retain(|expr| !matches!(expr, GrammarExpr::Ref(name) if name == JSON_INTEGER_RULE));
    }
    match alternatives.len() {
        0 => never(),
        1 => alternatives.pop().unwrap(),
        _ => GrammarExpr::Choice(alternatives),
    }
}

pub(crate) fn never() -> GrammarExpr {
    GrammarExpr::Choice(Vec::new())
}
