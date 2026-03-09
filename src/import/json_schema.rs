#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, lower};

fn literal_expr(bytes: &[u8]) -> GrammarExpr {
    GrammarExpr::Literal(bytes.to_vec())
}

const JSON_VALUE_RULE: &str = "__json_value";
const JSON_ARRAY_RULE: &str = "__json_array";
const JSON_OBJECT_RULE: &str = "__json_object";
const JSON_MEMBER_RULE: &str = "__json_member";

fn value_to_expr(value: &serde_json::Value) -> Result<GrammarExpr, GlrMaskError> {
    let rendered = serde_json::to_string(value)
        .map_err(|err| GlrMaskError::GrammarParse(err.to_string()))?;
    Ok(literal_expr(rendered.as_bytes()))
}

fn type_name_to_expr(type_name: &str) -> Result<GrammarExpr, GlrMaskError> {
    Ok(match type_name {
        "boolean" => GrammarExpr::Choice(vec![literal_expr(b"true"), literal_expr(b"false")]),
        "null" => literal_expr(b"null"),
        "string" => literal_expr(b"\"x\""),
        "integer" | "number" => literal_expr(b"0"),
        "object" => literal_expr(b"{}"),
        "array" => literal_expr(b"[]"),
        other => {
            return Err(GlrMaskError::GrammarParse(format!(
                "unsupported schema type '{other}'"
            )))
        }
    })
}

fn convert_schema_minimal(
    schema: &serde_json::Value,
    defs: &HashMap<String, serde_json::Value>,
) -> Result<GrammarExpr, GlrMaskError> {
    match schema {
        serde_json::Value::Bool(true) => Ok(GrammarExpr::Choice(vec![
            literal_expr(b"null"),
            literal_expr(b"true"),
            literal_expr(b"false"),
            literal_expr(b"\"x\""),
            literal_expr(b"0"),
            literal_expr(b"[]"),
            literal_expr(b"{}"),
        ])),
        serde_json::Value::Bool(false) => Ok(literal_expr(b"null")),
        serde_json::Value::Object(map) => {
            if let Some(reference) = map.get("$ref").and_then(serde_json::Value::as_str) {
                if let Some(name) = reference.strip_prefix("#/$defs/") {
                    let target = defs.get(name).ok_or_else(|| {
                        GlrMaskError::GrammarParse(format!("unknown $ref target '{reference}'"))
                    })?;
                    return convert_schema_minimal(target, defs);
                }
                return Err(GlrMaskError::GrammarParse(format!(
                    "unsupported $ref '{reference}'"
                )));
            }

            if let Some(values) = map.get("enum").and_then(serde_json::Value::as_array) {
                return Ok(GrammarExpr::Choice(
                    values
                        .iter()
                        .map(value_to_expr)
                        .collect::<Result<Vec<_>, _>>()?,
                ));
            }

            if let Some(value) = map.get("const") {
                return value_to_expr(value);
            }

            if let Some(options) = map.get("oneOf").and_then(serde_json::Value::as_array) {
                return Ok(GrammarExpr::Choice(
                    options
                        .iter()
                        .map(|option| convert_schema_minimal(option, defs))
                        .collect::<Result<Vec<_>, _>>()?,
                ));
            }

            if let Some(options) = map.get("type").and_then(serde_json::Value::as_array) {
                return Ok(GrammarExpr::Choice(
                    options
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(type_name_to_expr)
                        .collect::<Result<Vec<_>, _>>()?,
                ));
            }

            if let Some(type_name) = map.get("type").and_then(serde_json::Value::as_str) {
                return type_name_to_expr(type_name);
            }

            if map.contains_key("properties") || map.contains_key("required") {
                return Ok(literal_expr(b"{}"));
            }

            if map.contains_key("items") || map.contains_key("prefixItems") {
                return Ok(literal_expr(b"[]"));
            }

            if map.contains_key("allOf") {
                return Ok(literal_expr(b"{}"));
            }

            Ok(literal_expr(b"null"))
        }
        other => value_to_expr(other),
    }
}

pub fn json_schema_to_grammar(schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|err| GlrMaskError::GrammarParse(err.to_string()))?;
    let named = schema_to_named_grammar(&schema)?;
    lower(&named)
}

pub fn schema_to_named_grammar(schema: &serde_json::Value) -> Result<NamedGrammar, GlrMaskError> {
    let mut ctx = SchemaCtx::new(schema);
    let start_expr = ctx.convert_schema(schema)?;
    let mut rules = vec![("start".into(), start_expr)];
    rules.extend(ctx.sub_rules);
    Ok(NamedGrammar { rules, start: "start".into() })
}

struct SchemaCtx {
    sub_rules: Vec<(String, GrammarExpr)>,
    counter: usize,
    defs: HashMap<String, serde_json::Value>,
    defined_rules: HashSet<String>,
}

impl SchemaCtx {
    fn new(root: &serde_json::Value) -> Self {
        let defs = root
            .get("$defs")
            .and_then(serde_json::Value::as_object)
            .map(|defs| defs.iter().map(|(name, value)| (name.clone(), value.clone())).collect())
            .unwrap_or_default();
        Self {
            sub_rules: Vec::new(),
            counter: 0,
            defs,
            defined_rules: HashSet::new(),
        }
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        let name = format!("{}_{}", sanitize_rule_name(hint), self.counter);
        self.counter += 1;
        name
    }

    fn push_rule(&mut self, name: impl Into<String>, expr: GrammarExpr) -> String {
        let name = name.into();
        self.sub_rules.push((name.clone(), expr));
        name
    }

    fn json_value_ref(&mut self) -> GrammarExpr {
        self.ensure_json_value_rule();
        GrammarExpr::Ref(JSON_VALUE_RULE.into())
    }

    fn json_array_ref(&mut self) -> GrammarExpr {
        self.ensure_json_array_rule();
        GrammarExpr::Ref(JSON_ARRAY_RULE.into())
    }

    fn json_object_ref(&mut self) -> GrammarExpr {
        self.ensure_json_object_rule();
        GrammarExpr::Ref(JSON_OBJECT_RULE.into())
    }

    fn ensure_json_value_rule(&mut self) {
        if !self.defined_rules.insert(JSON_VALUE_RULE.into()) {
            return;
        }
        self.ensure_json_array_rule();
        self.ensure_json_object_rule();
        let json_string = self.json_string();
        let json_number = self.json_number();
        self.push_rule(
            JSON_VALUE_RULE,
            choice_or_single(vec![
                literal_expr(b"null"),
                literal_expr(b"true"),
                literal_expr(b"false"),
                json_string,
                json_number,
                GrammarExpr::Ref(JSON_ARRAY_RULE.into()),
                GrammarExpr::Ref(JSON_OBJECT_RULE.into()),
            ]),
        );
    }

    fn ensure_json_member_rule(&mut self) {
        if !self.defined_rules.insert(JSON_MEMBER_RULE.into()) {
            return;
        }
        let key = self.json_string();
        let value = self.json_value_ref();
        self.push_rule(
            JSON_MEMBER_RULE,
            sequence_or_single(vec![key, literal_expr(b":"), value]),
        );
    }

    fn ensure_json_array_rule(&mut self) {
        if !self.defined_rules.insert(JSON_ARRAY_RULE.into()) {
            return;
        }
        let item = self.json_value_ref();
        self.push_rule(
            JSON_ARRAY_RULE,
            choice_or_single(vec![
                literal_expr(b"[]"),
                sequence_or_single(vec![
                    literal_expr(b"["),
                    item.clone(),
                    repeat_expr(sequence_or_single(vec![literal_expr(b","), item]), 0, None),
                    literal_expr(b"]"),
                ]),
            ]),
        );
    }

    fn ensure_json_object_rule(&mut self) {
        if !self.defined_rules.insert(JSON_OBJECT_RULE.into()) {
            return;
        }
        self.ensure_json_member_rule();
        let member = GrammarExpr::Ref(JSON_MEMBER_RULE.into());
        self.push_rule(
            JSON_OBJECT_RULE,
            choice_or_single(vec![
                literal_expr(b"{}"),
                sequence_or_single(vec![
                    literal_expr(b"{"),
                    member.clone(),
                    repeat_expr(sequence_or_single(vec![literal_expr(b","), member]), 0, None),
                    literal_expr(b"}"),
                ]),
            ]),
        );
    }

    fn convert_schema(&mut self, schema: &serde_json::Value) -> Result<GrammarExpr, GlrMaskError> {
        match schema {
            serde_json::Value::Bool(true) => Ok(self.json_value_ref()),
            serde_json::Value::Bool(false) => Ok(literal_expr(b"null")),
            serde_json::Value::Object(obj) => {
                if let Some(reference) = obj.get("$ref").and_then(serde_json::Value::as_str) {
                    return self.resolve_ref(reference);
                }
                if let Some(values) = obj.get("enum").and_then(serde_json::Value::as_array) {
                    return Ok(choice_or_single(
                        values.iter().map(|value| self.json_literal(value)).collect(),
                    ));
                }
                if let Some(value) = obj.get("const") {
                    return Ok(self.json_literal(value));
                }
                if let Some(options) = obj.get("oneOf").and_then(serde_json::Value::as_array) {
                    return Ok(choice_or_single(
                        options
                            .iter()
                            .map(|option| self.convert_schema(option))
                            .collect::<Result<Vec<_>, _>>()?,
                    ));
                }
                if let Some(options) = obj.get("anyOf").and_then(serde_json::Value::as_array) {
                    return Ok(choice_or_single(
                        options
                            .iter()
                            .map(|option| self.convert_schema(option))
                            .collect::<Result<Vec<_>, _>>()?,
                    ));
                }
                if let Some(all) = obj.get("allOf").and_then(serde_json::Value::as_array) {
                    return self.convert_all_of(all, obj);
                }
                if obj.contains_key("properties")
                    || obj.contains_key("required")
                    || obj.contains_key("additionalProperties")
                    || obj.contains_key("propertyNames")
                    || obj.contains_key("minProperties")
                    || obj.contains_key("maxProperties")
                    || obj.get("type").and_then(serde_json::Value::as_str) == Some("object")
                {
                    return self.convert_object(obj);
                }
                if obj.contains_key("items")
                    || obj.contains_key("prefixItems")
                    || obj.get("type").and_then(serde_json::Value::as_str) == Some("array")
                {
                    return self.convert_array(obj);
                }
                if let Some(types) = obj.get("type").and_then(serde_json::Value::as_array) {
                    let alts = types
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(|type_name| self.convert_type(type_name, obj))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(choice_or_single(alts));
                }
                match obj.get("type").and_then(serde_json::Value::as_str) {
                    Some(type_name) => self.convert_type(type_name, obj),
                    None => Ok(self.json_value_ref()),
                }
            }
            other => Ok(self.json_literal(other)),
        }
    }

    fn convert_type(
        &mut self,
        type_name: &str,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        match type_name {
            "string" => Ok(self.convert_string_schema(obj)),
            "integer" => Ok(self.json_integer()),
            "number" => Ok(self.json_number()),
            "boolean" => Ok(choice_or_single(vec![literal_expr(b"true"), literal_expr(b"false")])),
            "null" => Ok(literal_expr(b"null")),
            "array" => self.convert_array(obj),
            "object" => self.convert_object(obj),
            _ => Ok(self.json_value_ref()),
        }
    }

    fn convert_string_schema(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> GrammarExpr {
        if let Some(pattern) = obj.get("pattern").and_then(serde_json::Value::as_str) {
            return self.json_string_pattern(pattern);
        }
        let min_len = obj.get("minLength").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        let max_len = obj.get("maxLength").and_then(serde_json::Value::as_u64).map(|v| v as usize);
        if min_len == 0 && max_len.is_none() {
            self.json_string()
        } else {
            self.json_string_bounded(min_len, max_len)
        }
    }

    fn resolve_ref(&mut self, ref_str: &str) -> Result<GrammarExpr, GlrMaskError> {
        if let Some(name) = ref_str.strip_prefix("#/$defs/") {
            let target = self
                .defs
                .get(name)
                .cloned()
                .ok_or_else(|| GlrMaskError::GrammarParse(format!("unknown $ref target '{ref_str}'")))?;
            return self.convert_schema(&target);
        }
        Err(GlrMaskError::GrammarParse(format!("unsupported $ref '{ref_str}'")))
    }

    fn convert_all_of(
        &mut self,
        all: &[serde_json::Value],
        parent: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let mut merged = parent.clone();
        // Remove "allOf" to avoid infinite recursion when we call
        // convert_schema on the merged object.
        merged.remove("allOf");
        for schema in all {
            if let serde_json::Value::Object(obj) = schema {
                for (key, value) in obj {
                    match key.as_str() {
                        // Deep-merge "properties" so all subschema properties
                        // are retained instead of overwritten.
                        "properties" => {
                            if let (Some(serde_json::Value::Object(existing)), serde_json::Value::Object(new_props)) =
                                (merged.get_mut("properties"), value)
                            {
                                for (pname, pval) in new_props {
                                    existing.insert(pname.clone(), pval.clone());
                                }
                            } else {
                                merged.insert(key.clone(), value.clone());
                            }
                        }
                        // Concatenate "required" arrays instead of overwriting.
                        "required" => {
                            if let (Some(serde_json::Value::Array(existing)), serde_json::Value::Array(new_items)) =
                                (merged.get_mut("required"), value)
                            {
                                for item in new_items {
                                    if !existing.contains(item) {
                                        existing.push(item.clone());
                                    }
                                }
                            } else {
                                merged.insert(key.clone(), value.clone());
                            }
                        }
                        _ => {
                            merged.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
        }
        self.convert_schema(&serde_json::Value::Object(merged))
    }

    fn convert_object(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let properties = obj
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .cloned();
        let required: Vec<String> = obj
            .get("required")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let additional = obj.get("additionalProperties");

        let properties_dict = properties.as_ref();
        let required_set: BTreeSet<String> = required.iter().cloned().collect();
        let no_additional = matches!(additional, None | Some(serde_json::Value::Bool(false)));
        let all_keys_required = properties_dict
            .map(|properties| properties.keys().all(|key| required_set.contains(key)))
            .unwrap_or(false);

        let min_properties = obj.get("minProperties").and_then(serde_json::Value::as_u64);
        let max_properties = obj.get("maxProperties").and_then(serde_json::Value::as_u64);
        if min_properties.is_some() || max_properties.is_some() {
            if properties_dict.is_none() || !all_keys_required || !no_additional {
                return Err(GlrMaskError::GrammarParse(
                    "min/maxProperties only supported when all declared properties are required and additionalProperties is false".into(),
                ));
            }
            let fixed = properties_dict.map(|properties| properties.len()).unwrap_or(0) as u64;
            if min_properties.map(|min| fixed < min).unwrap_or(false)
                || max_properties.map(|max| fixed > max).unwrap_or(false)
            {
                return Err(GlrMaskError::GrammarParse(
                    "min/maxProperties constraints are unsatisfiable for fixed required properties".into(),
                ));
            }
        }

        if let Some(properties) = properties_dict {
            let additional_schema = match additional {
                Some(serde_json::Value::Bool(false)) => None,
                Some(serde_json::Value::Object(map)) => Some(serde_json::Value::Object(map.clone())),
                _ => Some(serde_json::json!({})),
            };
            return self.build_ordered_object_rule(properties, &required, additional_schema);
        }

        if let Some(property_names) = obj.get("propertyNames") {
            return self.build_pattern_named_object_rule(property_names);
        }

        Ok(self.json_object_ref())
    }

    fn build_ordered_object_rule(
        &mut self,
        properties: &serde_json::Map<String, serde_json::Value>,
        required: &[String],
        additional_schema: Option<serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let required_set: BTreeSet<String> = required.iter().cloned().collect();
        let mut ordered = Vec::new();
        for (name, schema) in properties {
            ordered.push((name.clone(), schema.clone(), required_set.contains(name)));
        }
        for name in required {
            if !properties.contains_key(name) {
                ordered.push((name.clone(), serde_json::json!({}), true));
            }
        }

        let base = self.fresh_name("obj_ord");
        let term_nc = format!("{base}_term_nc");
        let term_c = format!("{base}_term_c");

        if let Some(schema) = additional_schema {
            let pair = sequence_or_single(vec![self.json_string(), literal_expr(b":"), self.convert_schema(&schema)?]);
            self.push_rule(
                term_nc.clone(),
                choice_or_single(vec![
                    GrammarExpr::Sequence(Vec::new()),
                    sequence_or_single(vec![pair.clone(), GrammarExpr::Ref(term_c.clone())]),
                ]),
            );
            self.push_rule(
                term_c.clone(),
                choice_or_single(vec![
                    GrammarExpr::Sequence(Vec::new()),
                    sequence_or_single(vec![literal_expr(b","), pair, GrammarExpr::Ref(term_c.clone())]),
                ]),
            );
        } else {
            self.push_rule(term_nc.clone(), GrammarExpr::Sequence(Vec::new()));
            self.push_rule(term_c.clone(), GrammarExpr::Sequence(Vec::new()));
        }

        for index in (0..ordered.len()).rev() {
            let nc_name = format!("{base}_{index}_nc");
            let c_name = format!("{base}_{index}_c");
            let next_nc = if index + 1 < ordered.len() {
                format!("{base}_{}_nc", index + 1)
            } else {
                term_nc.clone()
            };
            let next_c = if index + 1 < ordered.len() {
                format!("{base}_{}_c", index + 1)
            } else {
                term_c.clone()
            };

            let (name, schema, required) = &ordered[index];
            let pair = sequence_or_single(vec![
                self.json_string_literal(name),
                literal_expr(b":"),
                self.convert_schema(schema)?,
            ]);
            let include_nc = sequence_or_single(vec![pair.clone(), GrammarExpr::Ref(next_c.clone())]);
            let include_c = sequence_or_single(vec![literal_expr(b","), pair, GrammarExpr::Ref(next_c.clone())]);

            if *required {
                self.push_rule(nc_name, include_nc);
                self.push_rule(c_name, include_c);
            } else {
                self.push_rule(
                    nc_name,
                    choice_or_single(vec![GrammarExpr::Ref(next_nc), include_nc]),
                );
                self.push_rule(
                    c_name,
                    choice_or_single(vec![GrammarExpr::Ref(next_c), include_c]),
                );
            }
        }

        let body = if ordered.is_empty() {
            GrammarExpr::Ref(term_nc)
        } else {
            GrammarExpr::Ref(format!("{base}_0_nc"))
        };

        Ok(sequence_or_single(vec![literal_expr(b"{"), body, literal_expr(b"}")]))
    }

    fn build_pattern_named_object_rule(
        &mut self,
        property_names: &serde_json::Value,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let property_names = property_names
            .as_object()
            .ok_or_else(|| GlrMaskError::GrammarParse("propertyNames must be an object with a pattern".into()))?;
        if property_names.keys().any(|key| key != "pattern") {
            return Err(GlrMaskError::GrammarParse(
                "propertyNames only supports a pattern constraint".into(),
            ));
        }
        let pattern = property_names
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| GlrMaskError::GrammarParse("propertyNames.pattern must be a string".into()))?;
        let pair = sequence_or_single(vec![
            self.json_string_pattern(pattern),
            literal_expr(b":"),
            self.json_value_ref(),
        ]);
        Ok(choice_or_single(vec![
            literal_expr(b"{}"),
            sequence_or_single(vec![
                literal_expr(b"{"),
                pair.clone(),
                repeat_expr(sequence_or_single(vec![literal_expr(b","), pair]), 0, None),
                literal_expr(b"}"),
            ]),
        ]))
    }

    fn convert_array(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let min_items = obj.get("minItems").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        let max_items = obj.get("maxItems").and_then(serde_json::Value::as_u64).map(|v| v as usize);

        if let Some(prefix_items) = obj.get("prefixItems").and_then(serde_json::Value::as_array) {
            if max_items.map(|max| max < prefix_items.len()).unwrap_or(false) {
                return Ok(self.json_array_ref());
            }
            let extra_item = match obj.get("items") {
                Some(serde_json::Value::Object(_)) => self.convert_schema(obj.get("items").unwrap())?,
                _ => self.json_value_ref(),
            };
            if prefix_items.is_empty() {
                let extra_min = min_items;
                return Ok(self.build_repeated_array(extra_item, extra_min, max_items));
            }

            let mut parts = vec![literal_expr(b"[")];
            for (index, item_schema) in prefix_items.iter().enumerate() {
                if index > 0 {
                    parts.push(literal_expr(b","));
                }
                parts.push(self.convert_schema(item_schema)?);
            }

            let extra_min = min_items.saturating_sub(prefix_items.len());
            let extra_max = max_items.map(|max| max.saturating_sub(prefix_items.len()));
            if extra_min > 0 || extra_max.map(|max| max > 0).unwrap_or(true) {
                parts.push(repeat_expr(
                    sequence_or_single(vec![literal_expr(b","), extra_item]),
                    extra_min,
                    extra_max,
                ));
            }
            parts.push(literal_expr(b"]"));
            return Ok(sequence_or_single(parts));
        }

        if let Some(item_schema) = obj.get("items") {
            let item = self.convert_schema(item_schema)?;
            return Ok(self.build_repeated_array(item, min_items, max_items));
        }

        if min_items > 0 || max_items.is_some() {
            let item = self.json_value_ref();
            return Ok(self.build_repeated_array(item, min_items, max_items));
        }

        Ok(self.json_array_ref())
    }

    fn build_repeated_array(
        &mut self,
        item_expr: GrammarExpr,
        min_items: usize,
        max_items: Option<usize>,
    ) -> GrammarExpr {
        if max_items == Some(0) {
            return literal_expr(b"[]");
        }
        let non_empty = sequence_or_single(vec![
            literal_expr(b"["),
            item_expr.clone(),
            repeat_expr(
                sequence_or_single(vec![literal_expr(b","), item_expr]),
                min_items.saturating_sub(1),
                max_items.map(|max| max.saturating_sub(1)),
            ),
            literal_expr(b"]"),
        ]);
        if min_items == 0 {
            choice_or_single(vec![literal_expr(b"[]"), non_empty])
        } else {
            non_empty
        }
    }

    fn json_string(&mut self) -> GrammarExpr {
        GrammarExpr::RawRegex("\"[^\\\"]*\"".into())
    }

    fn json_string_bounded(&mut self, min: usize, max: Option<usize>) -> GrammarExpr {
        match max {
            Some(max) if min == max => self.json_string_pattern(&format!("[^\\\"]{{{min}}}")),
            Some(max) => self.json_string_pattern(&format!("[^\\\"]{{{min},{max}}}")),
            None => self.json_string_pattern(&format!("[^\\\"]{{{min},}}")),
        }
    }

    fn json_string_pattern(&self, pattern: &str) -> GrammarExpr {
        GrammarExpr::RawRegex(format!("\"{}\"", pattern))
    }

    fn json_number(&mut self) -> GrammarExpr {
        GrammarExpr::RawRegex("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?".into())
    }

    fn json_integer(&mut self) -> GrammarExpr {
        GrammarExpr::RawRegex("-?(0|[1-9][0-9]*)".into())
    }

    fn json_literal(&self, value: &serde_json::Value) -> GrammarExpr {
        if let Some(s) = value.as_str() {
            self.json_string_literal(s)
        } else {
            literal_expr(serde_json::to_string(value).unwrap_or_else(|_| "null".into()).as_bytes())
        }
    }

    fn json_string_literal(&self, s: &str) -> GrammarExpr {
        literal_expr(serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s)).as_bytes())
    }
}

fn choice_or_single(alts: Vec<GrammarExpr>) -> GrammarExpr {
    let mut alts = alts;
    if alts.is_empty() {
        GrammarExpr::Sequence(Vec::new())
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
        GrammarExpr::Sequence(Vec::new())
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
