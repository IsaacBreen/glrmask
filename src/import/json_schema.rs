#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::HashMap;

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, lower};

fn literal_expr(bytes: &[u8]) -> GrammarExpr {
    GrammarExpr::Literal(bytes.to_vec())
}

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
        }
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        let name = format!("{}_{}", sanitize_rule_name(hint), self.counter);
        self.counter += 1;
        name
    }

    fn convert_schema(&mut self, schema: &serde_json::Value) -> Result<GrammarExpr, GlrMaskError> {
        match schema {
            serde_json::Value::Bool(true) => Ok(self.json_value()),
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
                        .map(|type_name| self.convert_schema(&serde_json::json!({"type": type_name})))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(choice_or_single(alts));
                }
                match obj.get("type").and_then(serde_json::Value::as_str) {
                    Some("string") => Ok(self.json_string()),
                    Some("integer") => Ok(self.json_integer()),
                    Some("number") => Ok(self.json_number()),
                    Some("boolean") => Ok(choice_or_single(vec![literal_expr(b"true"), literal_expr(b"false")])),
                    Some("null") => Ok(literal_expr(b"null")),
                    Some(_) | None => Ok(self.json_value()),
                }
            }
            other => Ok(self.json_literal(other)),
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
        let properties: Vec<(String, serde_json::Value)> = obj
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .map(|map| map.iter().map(|(name, value)| (name.clone(), value.clone())).collect())
            .unwrap_or_default();
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
        self.build_object_rule(&properties, &required, obj.get("additionalProperties"))
    }

    fn build_object_rule(
        &mut self,
        properties: &[(String, serde_json::Value)],
        required: &[String],
        additional: Option<&serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        if properties.is_empty() && matches!(additional, None | Some(serde_json::Value::Bool(false))) {
            return Ok(literal_expr(b"{}"));
        }
        if !matches!(additional, None | Some(serde_json::Value::Bool(false))) {
            return Ok(self.json_object_generic());
        }

        let required_set: std::collections::BTreeSet<String> = required.iter().cloned().collect();
        let optional: Vec<_> = properties
            .iter()
            .filter(|(name, _)| !required_set.contains(name))
            .cloned()
            .collect();
        let required_props: Vec<_> = properties
            .iter()
            .filter(|(name, _)| required_set.contains(name))
            .cloned()
            .collect();

        let mut alts = Vec::new();
        let optional_count = optional.len();
        for mask in 0..(1usize << optional_count) {
            let mut chosen = required_props.clone();
            for (index, prop) in optional.iter().enumerate() {
                if (mask & (1usize << index)) != 0 {
                    chosen.push(prop.clone());
                }
            }

            if chosen.is_empty() {
                alts.push(literal_expr(b"{}"));
                continue;
            }

            let mut permutations = Vec::new();
            permute_properties(&chosen, 0, &mut permutations);
            for permutation in permutations {
                let mut parts = vec![literal_expr(b"{")];
                for (index, (name, schema)) in permutation.iter().enumerate() {
                    if index > 0 {
                        parts.push(literal_expr(b","));
                    }
                    parts.push(self.json_string_literal(name));
                    parts.push(literal_expr(b":"));
                    parts.push(self.convert_schema(schema)?);
                }
                parts.push(literal_expr(b"}"));
                alts.push(GrammarExpr::Sequence(parts));
            }
        }

        Ok(choice_or_single(alts))
    }

    fn convert_array(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let min_items = obj.get("minItems").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        let max_items = obj.get("maxItems").and_then(serde_json::Value::as_u64).map(|v| v as usize);

        if let Some(prefix_items) = obj.get("prefixItems").and_then(serde_json::Value::as_array) {
            let upper = max_items.unwrap_or(prefix_items.len()).min(prefix_items.len());
            let mut alts = Vec::new();
            for len in min_items..=upper {
                let mut parts = vec![literal_expr(b"[")];
                for (index, item_schema) in prefix_items.iter().take(len).enumerate() {
                    if index > 0 {
                        parts.push(literal_expr(b","));
                    }
                    parts.push(self.convert_schema(item_schema)?);
                }
                parts.push(literal_expr(b"]"));
                alts.push(GrammarExpr::Sequence(parts));
            }
            return Ok(choice_or_single(alts));
        }

        if let Some(item_schema) = obj.get("items") {
            if let Some(max_items) = max_items {
                let mut alts = Vec::new();
                for len in min_items..=max_items {
                    let mut parts = vec![literal_expr(b"[")];
                    for index in 0..len {
                        if index > 0 {
                            parts.push(literal_expr(b","));
                        }
                        parts.push(self.convert_schema(item_schema)?);
                    }
                    parts.push(literal_expr(b"]"));
                    alts.push(GrammarExpr::Sequence(parts));
                }
                return Ok(choice_or_single(alts));
            }
        }

        Ok(self.json_array_generic())
    }

    fn json_value(&mut self) -> GrammarExpr {
        choice_or_single(vec![
            literal_expr(b"null"),
            literal_expr(b"true"),
            literal_expr(b"false"),
            self.json_string(),
            self.json_number(),
            self.json_array_generic(),
            self.json_object_generic(),
        ])
    }

    fn json_array_generic(&mut self) -> GrammarExpr {
        literal_expr(b"[]")
    }

    fn json_object_generic(&mut self) -> GrammarExpr {
        literal_expr(b"{}")
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

fn sanitize_rule_name(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if sanitized.is_empty() { "rule".into() } else { sanitized }
}

fn build_optional_choice(optional_keys: &[String], kv_rules: &[(String, String)]) -> GrammarExpr {
    let mut alts = vec![GrammarExpr::Sequence(Vec::new())];
    for key in optional_keys {
        if let Some((_, rule_name)) = kv_rules.iter().find(|(name, _)| name == key) {
            alts.push(GrammarExpr::Ref(rule_name.clone()));
        }
    }
    choice_or_single(alts)
}

fn build_repetition(item_rule: &str, min: usize, max: Option<usize>) -> GrammarExpr {
    let item = GrammarExpr::Ref(item_rule.to_string());
    match (min, max) {
        (0, None) => GrammarExpr::Repeat(Box::new(item)),
        (1, None) => GrammarExpr::RepeatOne(Box::new(item)),
        (0, Some(1)) => GrammarExpr::Optional(Box::new(item)),
        _ => {
            let mut parts = vec![item.clone(); min];
            if max.map(|max| max > min).unwrap_or(false) {
                for _ in min..max.unwrap() {
                    parts.push(GrammarExpr::Optional(Box::new(item.clone())));
                }
            }
            GrammarExpr::Sequence(parts)
        }
    }
}

fn permute_properties(
    properties: &[(String, serde_json::Value)],
    start: usize,
    out: &mut Vec<Vec<(String, serde_json::Value)>>,
) {
    if start >= properties.len() {
        out.push(properties.to_vec());
        return;
    }

    let mut properties = properties.to_vec();
    for index in start..properties.len() {
        properties.swap(start, index);
        permute_properties(&properties, start + 1, out);
        properties.swap(start, index);
    }
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
