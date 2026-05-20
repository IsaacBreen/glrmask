use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule};

use super::ast::{
    Schema, SchemaAssertions, SchemaDocument, SchemaKind, SchemaType,
};
use super::config::JsonSchemaConfig;
use super::error::{ImportResult, SchemaImportError};

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
    definition_rules: BTreeMap<String, String>,
    definition_by_pointer: BTreeMap<String, &'a Schema>,
    used_rule_names: BTreeSet<String>,
    next_rule_id: usize,
}

impl<'a> Lowerer<'a> {
    fn new(document: &'a SchemaDocument, config: JsonSchemaConfig) -> Self {
        let mut definition_by_pointer = BTreeMap::new();
        for definition in &document.definitions {
            definition_by_pointer.insert(definition.pointer.clone(), &definition.schema);
        }
        definition_by_pointer.insert("#".to_string(), &document.root);

        let mut lowerer = Self {
            document,
            config,
            rules: Vec::new(),
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
        r#"[^\x00-\x1f\x7f"\\]|\\["\\bfnrt]"#.to_string()
    }

    pub(crate) fn lower_schema(&mut self, schema: &Schema) -> ImportResult<GrammarExpr> {
        match &schema.kind {
            SchemaKind::Any => Ok(r(JSON_VALUE_RULE)),
            SchemaKind::Never => Ok(never()),
            SchemaKind::Ref(pointer) => self.lower_ref(pointer),
            SchemaKind::Assertions(assertions) => self.lower_assertions(schema, assertions),
        }
    }

    fn lower_ref(&mut self, pointer: &str) -> ImportResult<GrammarExpr> {
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
            return self.lower_any_of(assertions);
        }
        if !assertions.one_of.is_empty() {
            return self.lower_one_of(assertions);
        }

        if let Some(value) = &assertions.const_value {
            return Ok(self.lower_json_literal(value));
        }
        if let Some(values) = &assertions.enum_values {
            return Ok(choice(values.iter().map(|value| self.lower_json_literal(value)).collect()));
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

    pub(crate) fn lower_json_literal(&mut self, value: &Value) -> GrammarExpr {
        match value {
            Value::String(text) => self.lower_string_literal(text),
            _ => lit_bytes(serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()).into_bytes()),
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

fn normalize_local_ref(pointer: &str) -> ImportResult<String> {
    if pointer == "#" {
        return Ok("#".to_string());
    }
    if pointer.starts_with("#/") {
        return Ok(pointer.to_string());
    }
    Err(SchemaImportError::new(format!(
        "only local JSON pointer $ref values are supported, got {pointer:?}"
    )))
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
    match alternatives.len() {
        0 => never(),
        1 => alternatives.pop().unwrap(),
        _ => GrammarExpr::Choice(alternatives),
    }
}

pub(crate) fn never() -> GrammarExpr {
    GrammarExpr::Choice(Vec::new())
}
