use std::collections::{BTreeMap, BTreeSet};

use regex::escape as regex_escape;
use serde_json::Value;

use crate::grammar::expr_nfa::ExprNFA;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, Quantifier};

use super::ast::{
    AdditionalProperties, ArraySchema, NumberSchema, ObjectSchema, Schema, SchemaAssertions,
    SchemaDocument, SchemaKind, SchemaType,
};
use super::config::JsonSchemaConfig;
use super::error::{ImportResult, SchemaImportError};
use super::string::{property_name_matches_pattern, string_value_satisfies_schema};

pub(crate) const JSON_VALUE_RULE: &str = "json_value";
pub(crate) const JSON_OBJECT_RULE: &str = "json_object";
pub(crate) const JSON_ARRAY_RULE: &str = "json_array";
pub(crate) const JSON_STRING_RULE: &str = "JSON_STRING";
pub(crate) const JSON_KEY_STRING_RULE: &str = "JSON_KEY_STRING";
pub(crate) const JSON_ADDITIONAL_KEY_STRING_RULE: &str = "JSON_ADDITIONAL_KEY_STRING";
pub(crate) const JSON_STRING_CHAR_RULE: &str = "JSON_STRING_CHAR";
pub(crate) const JSON_STRING_PATTERN_DOT_CHAR_RULE: &str = "JSON_STRING_PATTERN_DOT_CHAR";
pub(crate) const JSON_KEY_STRING_CHAR_RULE: &str = "JSON_KEY_STRING_CHAR";
pub(crate) const JSON_ADDITIONAL_KEY_STRING_CHAR_RULE: &str = "JSON_ADDITIONAL_KEY_STRING_CHAR";
pub(crate) const JSON_ITEM_SEPARATOR_RULE: &str = "JSON_ITEM_SEPARATOR";
pub(crate) const JSON_KEY_SEPARATOR_RULE: &str = "JSON_KEY_SEPARATOR";
pub(crate) const JSON_INTEGER_RULE: &str = "JSON_INTEGER";
pub(crate) const JSON_NUMBER_RULE: &str = "JSON_NUMBER";
pub(crate) const JSON_BOOL_RULE: &str = "JSON_BOOL";
pub(crate) const JSON_NULL_RULE: &str = "JSON_NULL";
pub(crate) const JSON_SEPARATOR_WS_REGEX: &str = r#" "#;
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
    pub(crate) shared_string_exact_open_rules: BTreeMap<usize, String>,
    pub(crate) shared_string_upto_wrapped_rules: BTreeMap<usize, String>,
    pub(crate) shared_ap_literal_keys: BTreeSet<String>,
    pub(crate) shared_ap_patterns: Vec<String>,
    pub(crate) shared_ap_base_rule: Option<String>,
    pub(crate) shared_ap_excluded_rule: Option<String>,
    pub(crate) shared_additional_key_colon_local_rules: BTreeMap<(Vec<String>, Vec<String>), String>,
    pub(crate) shared_ap_pattern_rules: BTreeMap<String, String>,
    pub(crate) shared_pattern_overlap_keys: BTreeMap<String, Vec<String>>,
    pub(crate) shared_pattern_overlap_literal_rules: BTreeMap<String, String>,
    pub(crate) shared_pattern_appearance_rules: BTreeMap<(String, Vec<String>), String>,
    definition_rules: BTreeMap<String, String>,
    definition_by_pointer: BTreeMap<String, &'a Schema>,
    used_rule_names: BTreeSet<String>,
    next_rule_id: usize,
}

fn quoted_repeated_char_rule_expr(char_rule: &str) -> GrammarExpr {
    seq(vec![
        lit("\""),
        GrammarExpr::Quantified(Box::new(r(char_rule)), Quantifier::ZeroPlus),
        lit("\""),
    ])
}

impl<'a> Lowerer<'a> {
    pub(crate) fn llguidance_compat_enabled(&self) -> bool {
        self.config.llguidance_compat
    }

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
            shared_string_exact_open_rules: BTreeMap::new(),
            shared_string_upto_wrapped_rules: BTreeMap::new(),
            shared_ap_literal_keys,
            shared_ap_patterns,
            shared_ap_base_rule: None,
            shared_ap_excluded_rule: None,
            shared_additional_key_colon_local_rules: BTreeMap::new(),
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
        simplify_terminal_rules(&mut self.rules);
        Ok(NamedGrammar { rules: self.rules, start: "start".to_string(), ignore: None })
    }

    fn install_json_builtins(&mut self) {
        let mode = super::string::json_string_compat_mode();
        let value_string_char = super::string::json_string_body_char_regex_in_mode(
            mode,
            super::string::JsonStringContext::Value,
        );
        let key_string_char = super::string::json_string_body_char_regex_in_mode(
            mode,
            super::string::JsonStringContext::KeyStrict,
        );
        let additional_key_string_char = super::string::json_string_body_char_regex_in_mode(
            mode,
            super::string::JsonStringContext::KeyAdditional,
        );
        self.add_internal_terminal_rule(
            JSON_STRING_CHAR_RULE,
            GrammarExpr::RawRegex(value_string_char.to_string()),
        );
        self.add_terminal_rule(
            JSON_STRING_RULE,
            quoted_repeated_char_rule_expr(JSON_STRING_CHAR_RULE),
        );
        if mode == super::string::JsonStringCompatMode::LlGuidanceNative {
            self.add_internal_terminal_rule(
                JSON_KEY_STRING_CHAR_RULE,
                GrammarExpr::RawRegex(key_string_char.to_string()),
            );
            self.add_terminal_rule(
                JSON_KEY_STRING_RULE,
                quoted_repeated_char_rule_expr(JSON_KEY_STRING_CHAR_RULE),
            );
            self.add_internal_terminal_rule(
                JSON_ADDITIONAL_KEY_STRING_CHAR_RULE,
                GrammarExpr::RawRegex(additional_key_string_char.to_string()),
            );
            self.add_terminal_rule(
                JSON_ADDITIONAL_KEY_STRING_RULE,
                quoted_repeated_char_rule_expr(JSON_ADDITIONAL_KEY_STRING_CHAR_RULE),
            );
        }
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
                GrammarExpr::Quantified(Box::new(seq(vec![
                    r(JSON_VALUE_RULE),
                    GrammarExpr::Quantified(Box::new(array_item_tail), Quantifier::ZeroPlus),
                ])), Quantifier::Optional),
                lit("]"),
            ]),
        );

        let object_entry = seq(vec![r(json_key_string_rule()), r(JSON_KEY_SEPARATOR_RULE), r(JSON_VALUE_RULE)]);
        let object_tail = seq(vec![r(JSON_ITEM_SEPARATOR_RULE), object_entry.clone()]);
        self.add_nonterminal_rule(
            JSON_OBJECT_RULE,
            seq(vec![
                lit("{"),
                GrammarExpr::Quantified(Box::new(seq(vec![
                    object_entry,
                    GrammarExpr::Quantified(Box::new(object_tail), Quantifier::ZeroPlus),
                ])), Quantifier::Optional),
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
            "," | ":" => format!("(?:{separator}{JSON_SEPARATOR_WS_REGEX})"),
            _ => format!("(?:{separator})"),
        }
    }

    pub(crate) fn json_string_char_regex(&self) -> String {
        super::string::json_string_body_char_regex().to_string()
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
            if self.json_literal_satisfies_assertions(value, assertions)? {
                return Ok(self.lower_json_literal(value));
            }
            return Ok(never());
        }
        if let Some(values) = &assertions.enum_values {
            let values = values
                .iter()
                .filter_map(|value| match self.json_literal_satisfies_assertions(value, assertions) {
                    Ok(true) => Some(Ok(value)),
                    Ok(false) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect::<ImportResult<Vec<_>>>()?;
            if let Some(encoded_literals) = large_string_enum_regex_literals(assertions, &values)? {
                return Ok(GrammarExpr::RawRegex(string_enum_regex(&encoded_literals)));
            }
            if let Some(expr) = factored_small_string_enum_expr(&values) {
                return Ok(expr);
            }
            return Ok(choice(values.into_iter().map(|value| self.lower_json_literal(value)).collect()));
        }

        if assertions.types.is_none() {
            let inferred = self.inferred_constrained_types(assertions);
            if inferred.len() == 1 {
                if self.llguidance_compat_enabled() {
                    if inferred[0] == SchemaType::Object {
                        return self.lower_untyped_single_family_assertions(inferred[0], assertions);
                    }
                    return self.lower_for_type(inferred[0], assertions);
                }
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

    fn json_literal_satisfies_schema(&self, value: &Value, schema: &Schema) -> ImportResult<bool> {
        let mut ref_stack = BTreeSet::new();
        self.json_literal_satisfies_schema_inner(value, schema, &mut ref_stack)
    }

    fn json_literal_satisfies_schema_inner(
        &self,
        value: &Value,
        schema: &Schema,
        ref_stack: &mut BTreeSet<String>,
    ) -> ImportResult<bool> {
        match &schema.kind {
            SchemaKind::Any => Ok(true),
            SchemaKind::Never => Ok(false),
            SchemaKind::Ref(pointer) => {
                let normalized = normalize_local_ref(pointer)?;
                if !ref_stack.insert(normalized.clone()) {
                    // Recursive schemas can be satisfied by finite literals, but
                    // proving that here is not needed for enum/const pruning.
                    // Avoid looping and let the surrounding non-recursive
                    // assertions do the remaining filtering.
                    return Ok(true);
                }
                let target = self.resolve_ref_target(pointer)?;
                let result = self.json_literal_satisfies_schema_inner(value, target, ref_stack);
                ref_stack.remove(&normalized);
                result
            }
            SchemaKind::Assertions(assertions) => {
                self.json_literal_satisfies_assertions_inner(value, assertions, ref_stack)
            }
        }
    }

    fn json_literal_satisfies_assertions(
        &self,
        value: &Value,
        assertions: &SchemaAssertions,
    ) -> ImportResult<bool> {
        let mut ref_stack = BTreeSet::new();
        self.json_literal_satisfies_assertions_inner(value, assertions, &mut ref_stack)
    }

    fn json_literal_satisfies_assertions_inner(
        &self,
        value: &Value,
        assertions: &SchemaAssertions,
        ref_stack: &mut BTreeSet<String>,
    ) -> ImportResult<bool> {
        for branch in &assertions.all_of {
            if !self.json_literal_satisfies_schema_inner(value, branch, ref_stack)? {
                return Ok(false);
            }
        }
        if !assertions.any_of.is_empty() {
            let mut matched = false;
            for branch in &assertions.any_of {
                if self.json_literal_satisfies_schema_inner(value, branch, ref_stack)? {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Ok(false);
            }
        }
        if !assertions.one_of.is_empty() {
            let mut matches = 0usize;
            for branch in &assertions.one_of {
                if self.json_literal_satisfies_schema_inner(value, branch, ref_stack)? {
                    matches += 1;
                }
            }
            if matches != 1 {
                return Ok(false);
            }
        }
        if let Some(schema) = &assertions.not
            && self.json_literal_satisfies_schema_inner(value, schema, ref_stack)?
        {
            return Ok(false);
        }

        if let Some(const_value) = &assertions.const_value
            && value != const_value
        {
            return Ok(false);
        }
        if let Some(enum_values) = &assertions.enum_values
            && !enum_values.iter().any(|enum_value| enum_value == value)
        {
            return Ok(false);
        }
        if !json_literal_satisfies_declared_types(value, assertions.types.as_deref()) {
            return Ok(false);
        }
        if let Some(string) = &assertions.string
            && !string_value_satisfies_schema(value, string)?
        {
            return Ok(false);
        }
        if let Some(number) = &assertions.number
            && !number_value_satisfies_schema(value, number)
        {
            return Ok(false);
        }
        if let Some(object) = &assertions.object
            && !self.object_value_satisfies_schema(value, object, ref_stack)?
        {
            return Ok(false);
        }
        if let Some(array) = &assertions.array
            && !self.array_value_satisfies_schema(value, array, ref_stack)?
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn object_value_satisfies_schema(
        &self,
        value: &Value,
        schema: &ObjectSchema,
        ref_stack: &mut BTreeSet<String>,
    ) -> ImportResult<bool> {
        let Some(map) = value.as_object() else {
            return Ok(true);
        };
        if map.len() < schema.min_properties
            || schema.max_properties.is_some_and(|max| map.len() > max)
        {
            return Ok(false);
        }
        for required in &schema.required {
            if !map.contains_key(required) {
                return Ok(false);
            }
        }
        for (trigger, dependents) in &schema.property_dependencies {
            if map.contains_key(trigger) && dependents.iter().any(|dependent| !map.contains_key(dependent)) {
                return Ok(false);
            }
        }
        if let Some(property_names) = &schema.property_names {
            for key in map.keys() {
                if !self.json_literal_satisfies_schema_inner(
                    &Value::String(key.clone()),
                    property_names,
                    ref_stack,
                )? {
                    return Ok(false);
                }
            }
        }

        for (key, item) in map {
            let mut matched_known_property = false;
            if let Some(property) = schema.properties.iter().find(|property| property.name == key.as_str()) {
                matched_known_property = true;
                if !self.json_literal_satisfies_schema_inner(item, &property.schema, ref_stack)? {
                    return Ok(false);
                }
            }
            for pattern_property in &schema.pattern_properties {
                if property_name_matches_pattern(&pattern_property.pattern, key)? {
                    matched_known_property = true;
                    if !self.json_literal_satisfies_schema_inner(
                        item,
                        &pattern_property.schema,
                        ref_stack,
                    )? {
                        return Ok(false);
                    }
                }
            }
            if !matched_known_property {
                match &schema.additional_properties {
                    AdditionalProperties::AllowAny => {}
                    AdditionalProperties::Deny => return Ok(false),
                    AdditionalProperties::Schema(additional) => {
                        if !self.json_literal_satisfies_schema_inner(item, additional, ref_stack)? {
                            return Ok(false);
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    fn array_value_satisfies_schema(
        &self,
        value: &Value,
        schema: &ArraySchema,
        ref_stack: &mut BTreeSet<String>,
    ) -> ImportResult<bool> {
        let Some(items) = value.as_array() else {
            return Ok(true);
        };
        if items.len() < schema.min_items
            || schema.max_items.is_some_and(|max| items.len() > max)
        {
            return Ok(false);
        }
        for (index, item) in items.iter().enumerate() {
            let item_schema = schema
                .prefix_items
                .get(index)
                .unwrap_or(schema.items.as_ref());
            if !self.json_literal_satisfies_schema_inner(item, item_schema, ref_stack)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn add_nonterminal_rule(&mut self, name: &str, expr: GrammarExpr) {
        let expr = self.hoist_raw_regexes_out_of_expr_nfa_symbols(expr);
        self.used_rule_names.insert(name.to_string());
        self.rules.push(NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        });
    }

    fn hoist_raw_regexes_out_of_expr_nfa_symbols(&mut self, expr: GrammarExpr) -> GrammarExpr {
        match expr {
            GrammarExpr::ExprNFA(expr_nfa) => {
                let ExprNFA { nfa, symbols } = *expr_nfa;
                let symbols = symbols
                    .into_iter()
                    .map(|symbol| self.hoist_raw_regexes_out_of_expr_nfa_symbol(symbol))
                    .collect();
                GrammarExpr::ExprNFA(Box::new(ExprNFA::new(nfa, symbols)))
            }
            other => other,
        }
    }

    fn hoist_raw_regexes_out_of_expr_nfa_symbol(&mut self, expr: GrammarExpr) -> GrammarExpr {
        match expr {
            GrammarExpr::RawRegex(pattern) => {
                let rule_name = self.fresh_rule_name("json_fa_regex_symbol");
                self.add_terminal_rule(&rule_name, GrammarExpr::RawRegex(pattern));
                r(&rule_name)
            }
            GrammarExpr::Grouped(inner) => GrammarExpr::Grouped(Box::new(
                self.hoist_raw_regexes_out_of_expr_nfa_symbol(*inner),
            )),
            GrammarExpr::Sequence(items) => GrammarExpr::Sequence(
                items
                    .into_iter()
                    .map(|item| self.hoist_raw_regexes_out_of_expr_nfa_symbol(item))
                    .collect(),
            ),
            GrammarExpr::Choice(items) => GrammarExpr::Choice(
                items
                    .into_iter()
                    .map(|item| self.hoist_raw_regexes_out_of_expr_nfa_symbol(item))
                    .collect(),
            ),
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.hoist_raw_regexes_out_of_expr_nfa_symbol(*expr)),
                exclude: Box::new(self.hoist_raw_regexes_out_of_expr_nfa_symbol(*exclude)),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.hoist_raw_regexes_out_of_expr_nfa_symbol(*expr)),
                intersect: Box::new(self.hoist_raw_regexes_out_of_expr_nfa_symbol(*intersect)),
            },
            GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
                Box::new(self.hoist_raw_regexes_out_of_expr_nfa_symbol(*inner)),
                quantifier,
            ),
            GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
                GrammarExpr::SeparatedSequence {
                    items: items
                        .into_iter()
                        .map(|(item, quantifier)| {
                            (self.hoist_raw_regexes_out_of_expr_nfa_symbol(item), quantifier)
                        })
                        .collect(),
                    separator: Box::new(self.hoist_raw_regexes_out_of_expr_nfa_symbol(*separator)),
                    allow_empty,
                }
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                let ExprNFA { nfa, symbols } = *expr_nfa;
                let symbols = symbols
                    .into_iter()
                    .map(|symbol| self.hoist_raw_regexes_out_of_expr_nfa_symbol(symbol))
                    .collect();
                GrammarExpr::ExprNFA(Box::new(ExprNFA::new(nfa, symbols)))
            }
            other => other,
        }
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

fn json_literal_satisfies_declared_types(value: &Value, types: Option<&[SchemaType]>) -> bool {
    types.is_none_or(|types| {
        types
            .iter()
            .any(|schema_type| json_literal_has_type(value, *schema_type))
    })
}

fn json_literal_has_type(value: &Value, schema_type: SchemaType) -> bool {
    match schema_type {
        SchemaType::Null => value.is_null(),
        SchemaType::Boolean => value.is_boolean(),
        SchemaType::Object => value.is_object(),
        SchemaType::Array => value.is_array(),
        SchemaType::String => value.is_string(),
        SchemaType::Number => value.is_number(),
        SchemaType::Integer => value
            .as_number()
            .is_some_and(json_number_is_integer),
    }
}

fn number_value_satisfies_schema(value: &Value, schema: &NumberSchema) -> bool {
    let Some(number) = value.as_number() else {
        return true;
    };
    if schema.integer && !json_number_is_integer(number) {
        return false;
    }
    let Some(value) = number.as_f64() else {
        return false;
    };
    if let Some(minimum) = schema.minimum {
        if schema.exclusive_minimum {
            if value <= minimum {
                return false;
            }
        } else if value < minimum {
            return false;
        }
    }
    if let Some(maximum) = schema.maximum {
        if schema.exclusive_maximum {
            if value >= maximum {
                return false;
            }
        } else if value > maximum {
            return false;
        }
    }
    if let Some(multiple) = schema.multiple_of {
        let quotient = value / multiple;
        if (quotient - quotient.round()).abs() > 1e-9 {
            return false;
        }
    }
    true
}

fn json_number_is_integer(number: &serde_json::Number) -> bool {
    number.as_i64().is_some()
        || number.as_u64().is_some()
        || number
            .as_f64()
            .is_some_and(|value| value.is_finite() && value.fract() == 0.0)
}

fn large_string_enum_regex_literals(
    assertions: &SchemaAssertions,
    values: &[&Value],
) -> ImportResult<Option<Vec<String>>> {
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
    for value in values.iter().copied() {
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
    for value in values.iter().copied() {
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
            !matches!(object.additional_properties, AdditionalProperties::Deny)
            || !object.pattern_properties.is_empty();
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
        if let super::ast::AdditionalProperties::Schema(schema) = &object.additional_properties {
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


fn simplify_terminal_rules(rules: &mut [NamedRule]) {
    for rule in rules.iter_mut().filter(|rule| rule.is_terminal) {
        rule.expr = simplify_terminal_expr(rule.expr.clone());
    }
}

fn simplify_terminal_expr(expr: GrammarExpr) -> GrammarExpr {
    match expr {
        GrammarExpr::Grouped(inner) => GrammarExpr::Grouped(Box::new(simplify_terminal_expr(*inner))),
        GrammarExpr::Sequence(items) => simplify_terminal_sequence(items),
        GrammarExpr::Choice(alternatives) => choice(
            alternatives
                .into_iter()
                .map(simplify_terminal_expr)
                .collect(),
        ),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(simplify_terminal_expr(*expr)),
            exclude: Box::new(simplify_terminal_expr(*exclude)),
        },
        GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(simplify_terminal_expr(*expr)),
            intersect: Box::new(simplify_terminal_expr(*intersect)),
        },
        GrammarExpr::Quantified(inner, quantifier) => {
            GrammarExpr::Quantified(Box::new(simplify_terminal_expr(*inner)), quantifier)
        },
        GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
            GrammarExpr::SeparatedSequence {
                items: items
                    .into_iter()
                    .map(|(item, quantifier)| (simplify_terminal_expr(item), quantifier))
                    .collect(),
                separator: Box::new(simplify_terminal_expr(*separator)),
                allow_empty,
            }
        },
        GrammarExpr::RawRegex(regex) => {
            if let Some(byte) = fixed_ascii_regex_byte(&regex) {
                lit_bytes(vec![byte])
            } else {
                GrammarExpr::RawRegex(regex)
            }
        },
        other => other,
    }
}

fn simplify_terminal_sequence(items: Vec<GrammarExpr>) -> GrammarExpr {
    let mut simplified = Vec::new();
    let mut pending_literal = Vec::new();

    fn flush_pending(pending_literal: &mut Vec<u8>, simplified: &mut Vec<GrammarExpr>) {
        if !pending_literal.is_empty() {
            simplified.push(lit_bytes(std::mem::take(pending_literal)));
        }
    }

    for item in items.into_iter().map(simplify_terminal_expr) {
        match item {
            GrammarExpr::Epsilon => {}
            GrammarExpr::Sequence(nested) => {
                for nested_item in nested {
                    match nested_item {
                        GrammarExpr::Literal(mut bytes) => pending_literal.append(&mut bytes),
                        other => {
                            flush_pending(&mut pending_literal, &mut simplified);
                            simplified.push(other);
                        }
                    }
                }
            }
            GrammarExpr::Literal(mut bytes) => pending_literal.append(&mut bytes),
            other => {
                flush_pending(&mut pending_literal, &mut simplified);
                simplified.push(other);
            }
        }
    }

    flush_pending(&mut pending_literal, &mut simplified);
    seq(simplified)
}

fn fixed_ascii_regex_byte(regex: &str) -> Option<u8> {
    let bytes = regex.as_bytes();
    match bytes {
        [byte] if is_plain_fixed_ascii_regex_byte(*byte) => Some(*byte),
        [b'\\', byte] if is_escaped_fixed_ascii_regex_byte(*byte) => Some(*byte),
        _ => None,
    }
}

fn is_plain_fixed_ascii_regex_byte(byte: u8) -> bool {
    byte.is_ascii()
        && !byte.is_ascii_control()
        && !matches!(
            byte,
            b'\\' | b'.' | b'+' | b'*' | b'?' | b'(' | b')' | b'|' | b'[' | b']' | b'{' | b'}'
                | b'^' | b'$'
        )
}

fn is_escaped_fixed_ascii_regex_byte(byte: u8) -> bool {
    byte.is_ascii()
        && !byte.is_ascii_control()
        && !byte.is_ascii_alphanumeric()
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

/// Returns the rule name to use for JSON object keys.
/// In `LlGuidanceNative` compat mode, this is the strict key rule used by
/// literal `properties` and `patternProperties` key paths.
pub(crate) fn json_key_string_rule() -> &'static str {
    match super::string::json_string_compat_mode() {
        super::string::JsonStringCompatMode::JsonSchema => JSON_STRING_RULE,
        super::string::JsonStringCompatMode::LlGuidanceNative => JSON_KEY_STRING_RULE,
    }
}

/// Returns the rule name to use for additional/generic object keys.
pub(crate) fn json_additional_key_string_rule() -> &'static str {
    match super::string::json_string_compat_mode() {
        super::string::JsonStringCompatMode::JsonSchema => JSON_STRING_RULE,
        super::string::JsonStringCompatMode::LlGuidanceNative => JSON_ADDITIONAL_KEY_STRING_RULE,
    }
}
