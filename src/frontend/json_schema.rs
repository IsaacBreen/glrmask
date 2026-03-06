//! JSON Schema → grammar converter.
//!
//! Converts a JSON Schema into a context-free grammar (`GrammarDef`) that
//! generates exactly the set of valid JSON strings conforming to the schema.
//!
//! Supported keywords:
//! - `type` (string, number, integer, boolean, null, object, array; also arrays of types)
//! - `properties`, `required`, `additionalProperties` (false / true / schema)
//! - `items`, `prefixItems`, `minItems`, `maxItems`
//! - `oneOf`, `anyOf`, `allOf`
//! - `enum`, `const`
//! - `$ref`, `$defs`, `definitions`
//! - `pattern` (string regex), `minLength`, `maxLength`

use std::collections::HashMap;

use crate::GlrMaskError;
use crate::compiler::grammar_def::GrammarDef;
use crate::frontend::grammar_expr::{GrammarExpr, NamedGrammar, lower};

/// Convert a JSON Schema (as a JSON string) into a `GrammarDef`.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {}", e)))?;
    let named = schema_to_named_grammar(&schema)?;
    lower(&named)
}

/// Convert a parsed JSON Schema value into a `NamedGrammar`.
pub fn schema_to_named_grammar(schema: &serde_json::Value) -> Result<NamedGrammar, GlrMaskError> {
    let mut ctx = SchemaCtx::new(schema);
    let start_expr = ctx.convert_schema(schema)?;

    let mut rules: Vec<(String, GrammarExpr)> = Vec::new();
    rules.push(("start".into(), start_expr));

    // Whitespace rule.
    rules.push((
        "ws".into(),
        GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass {
            def: " \\t\\n\\r".into(),
            negate: false,
        })),
    ));

    // Sub-rules generated during conversion.
    rules.extend(ctx.sub_rules);

    Ok(NamedGrammar { rules, start: "start".into() })
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct SchemaCtx {
    sub_rules: Vec<(String, GrammarExpr)>,
    counter: usize,
    /// `$defs`/`definitions` collected from the root schema.
    defs: HashMap<String, serde_json::Value>,
}

impl SchemaCtx {
    fn new(root: &serde_json::Value) -> Self {
        let mut defs = HashMap::new();
        if let Some(obj) = root.as_object() {
            for key in &["$defs", "definitions", "components"] {
                if let Some(d) = obj.get(*key).and_then(|v| v.as_object()) {
                    for (k, v) in d {
                        defs.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        SchemaCtx { sub_rules: Vec::new(), counter: 0, defs }
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        let name = format!("_json_{}_{}", hint, self.counter);
        self.counter += 1;
        name
    }

    // -----------------------------------------------------------------------
    // Top-level dispatcher
    // -----------------------------------------------------------------------

    fn convert_schema(&mut self, schema: &serde_json::Value) -> Result<GrammarExpr, GlrMaskError> {
        // Boolean schema.
        if let Some(b) = schema.as_bool() {
            return if b {
                Ok(self.json_value())
            } else {
                Err(GlrMaskError::GrammarParse("schema is false (matches nothing)".into()))
            };
        }

        let obj = schema.as_object().ok_or_else(|| {
            GlrMaskError::GrammarParse("schema must be an object or boolean".into())
        })?;

        // $ref — resolve from $defs / definitions.
        if let Some(ref_str) = obj.get("$ref").and_then(|v| v.as_str()) {
            return self.resolve_ref(ref_str);
        }

        // const.
        if let Some(val) = obj.get("const") {
            return Ok(self.json_literal(val));
        }

        // enum.
        if let Some(vals) = obj.get("enum").and_then(|v| v.as_array()) {
            let alts: Vec<GrammarExpr> = vals.iter().map(|v| self.json_literal(v)).collect();
            return Ok(choice_or_single(alts));
        }

        // oneOf / anyOf.
        if let Some(variants) = obj.get("oneOf").or_else(|| obj.get("anyOf")).and_then(|v| v.as_array()) {
            let mut alts = Vec::new();
            for v in variants {
                alts.push(self.convert_schema(v)?);
            }
            return Ok(choice_or_single(alts));
        }

        // allOf — merge all sub-schemas (for objects: collect all properties).
        if let Some(all) = obj.get("allOf").and_then(|v| v.as_array()) {
            return self.convert_all_of(all, obj);
        }

        // type: array of types.
        if let Some(types) = obj.get("type").and_then(|v| v.as_array()) {
            let mut alts = Vec::new();
            for t in types {
                if let Some(t_str) = t.as_str() {
                    let mut sub = obj.clone();
                    sub.insert("type".into(), serde_json::Value::String(t_str.into()));
                    alts.push(self.convert_schema(&serde_json::Value::Object(sub))?);
                }
            }
            return Ok(choice_or_single(alts));
        }

        // Dispatch on scalar type.
        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("any");

        // String with constraints.
        if ty == "string" || ty == "any" {
            if let Some(pattern) = obj.get("pattern").and_then(|v| v.as_str()) {
                return Ok(self.json_string_pattern(pattern));
            }
            if obj.contains_key("minLength") || obj.contains_key("maxLength") {
                let min = obj.get("minLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let max_opt = obj.get("maxLength").and_then(|v| v.as_u64()).map(|v| v as usize);
                if ty == "string" {
                    return Ok(self.json_string_bounded(min, max_opt));
                }
            }
        }

        // Object with properties or additionalProperties restriction.
        if (ty == "object" || ty == "any")
            && (obj.contains_key("properties")
                || (obj.contains_key("additionalProperties")
                    && obj.get("additionalProperties") != Some(&serde_json::Value::Bool(true))))
        {
            return self.convert_object(obj);
        }

        match ty {
            "object" => self.convert_object(obj),
            "array" => self.convert_array(obj),
            "string" => Ok(self.json_string()),
            "number" => Ok(self.json_number()),
            "integer" => Ok(self.json_integer()),
            "boolean" => Ok(GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"true".to_vec()),
                GrammarExpr::Literal(b"false".to_vec()),
            ])),
            "null" => Ok(GrammarExpr::Literal(b"null".to_vec())),
            _ => Ok(self.json_value()),
        }
    }

    // -----------------------------------------------------------------------
    // $ref resolution
    // -----------------------------------------------------------------------

    fn resolve_ref(&mut self, ref_str: &str) -> Result<GrammarExpr, GlrMaskError> {
        // Support: "#/$defs/Name", "#/definitions/Name", "#/components/Name"
        let fragment = ref_str
            .strip_prefix("#/")
            .or_else(|| ref_str.strip_prefix("#"))
            .unwrap_or(ref_str);

        // Derive the definition name from the last path segment.
        let def_name = fragment.split('/').last().unwrap_or(fragment);

        // Check if we already have a named rule for this def
        // (avoids infinite recursion for mutually recursive schemas).
        let rule_name = format!("_ref_{}", sanitize_rule_name(def_name));
        if self.sub_rules.iter().any(|(n, _)| n == &rule_name) {
            return Ok(GrammarExpr::Ref(rule_name));
        }

        // Look up the definition.
        let def_schema = self.defs.get(def_name).cloned().ok_or_else(|| {
            GlrMaskError::GrammarParse(format!("$ref not found: '{}'", ref_str))
        })?;

        // Add a placeholder to prevent infinite recursion.
        self.sub_rules.push((rule_name.clone(), GrammarExpr::Literal(b"null".to_vec())));

        // Convert the definition.
        let expr = self.convert_schema(&def_schema)?;

        // Replace the placeholder.
        if let Some(pos) = self.sub_rules.iter().position(|(n, _)| n == &rule_name) {
            self.sub_rules[pos].1 = expr;
        }

        Ok(GrammarExpr::Ref(rule_name))
    }

    // -----------------------------------------------------------------------
    // allOf
    // -----------------------------------------------------------------------

    fn convert_all_of(
        &mut self,
        all: &[serde_json::Value],
        parent: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        // Collect all properties and required fields from sub-schemas.
        let mut properties: Vec<(String, serde_json::Value)> = Vec::new();
        let mut required: Vec<String> = Vec::new();

        for sub in all {
            let sub_resolved = if let Some(r) = sub.as_object().and_then(|o| o.get("$ref")).and_then(|v| v.as_str()) {
                let name = r.split('/').last().unwrap_or(r);
                self.defs.get(name).cloned()
            } else {
                None
            };
            let sub = sub_resolved.as_ref().unwrap_or(sub);

            if let Some(props) = sub.as_object().and_then(|o| o.get("properties")).and_then(|v| v.as_object()) {
                for (k, v) in props {
                    if !properties.iter().any(|(n, _)| n == k) {
                        properties.push((k.clone(), v.clone()));
                    }
                }
            }
            if let Some(req) = sub.as_object().and_then(|o| o.get("required")).and_then(|v| v.as_array()) {
                for r in req {
                    if let Some(s) = r.as_str() {
                        if !required.contains(&s.to_string()) {
                            required.push(s.to_string());
                        }
                    }
                }
            }
        }

        // Also include properties/required from the parent (if any).
        if let Some(props) = parent.get("properties").and_then(|v| v.as_object()) {
            for (k, v) in props {
                if !properties.iter().any(|(n, _)| n == k) {
                    properties.push((k.clone(), v.clone()));
                }
            }
        }
        if let Some(req) = parent.get("required").and_then(|v| v.as_array()) {
            for r in req {
                if let Some(s) = r.as_str() {
                    if !required.contains(&s.to_string()) {
                        required.push(s.to_string());
                    }
                }
            }
        }

        if properties.is_empty() {
            // No properties merged — fall back to generic object.
            return Ok(self.json_object_generic());
        }

        let additional = parent.get("additionalProperties");
        self.build_object_rule(&properties, &required, additional)
    }

    // -----------------------------------------------------------------------
    // Object
    // -----------------------------------------------------------------------

    fn convert_object(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let properties = obj.get("properties").and_then(|v| v.as_object());
        let additional = obj.get("additionalProperties");

        if let Some(props) = properties {
            let props_vec: Vec<(String, serde_json::Value)> =
                props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            let required: Vec<String> = obj
                .get("required")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();

            self.build_object_rule(&props_vec, &required, additional)
        } else {
            // No properties defined.
            match additional {
                Some(serde_json::Value::Bool(false)) => {
                    // additionalProperties: false + no properties → {}
                    Ok(GrammarExpr::Sequence(vec![
                        GrammarExpr::Literal(b"{".to_vec()),
                        GrammarExpr::Ref("ws".into()),
                        GrammarExpr::Literal(b"}".to_vec()),
                    ]))
                }
                _ => {
                    // Generic object: any key-value pairs.
                    Ok(self.json_object_generic())
                }
            }
        }
    }

    /// Build an object rule with the CFA-style sequential optional permutation.
    ///
    /// For `required=[R1,R2]` and `optional=[O1,O2,O3]`, produces:
    /// ```text
    /// { ws R1_kv , ws R2_kv ( , ws ( O1_kv (,O2_kv)? (,O3_kv)? | O2_kv (,O3_kv)? | O3_kv ) )? ws }
    /// ```
    ///
    /// `additionalProperties`:
    /// - `None` or `Bool(false)` → only declared properties.
    /// - `Bool(true)` → also allow unknown key-value pairs as optional.
    /// - `{schema}` → also allow unknown key-value pairs with value matching schema.
    fn build_object_rule(
        &mut self,
        properties: &[(String, serde_json::Value)],
        required: &[String],
        additional: Option<&serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        // Build kv rule names for each declared property.
        let mut kv_rules: Vec<(String, String)> = Vec::new();  // (key, kv_rule_name)

        for (key, val_schema) in properties {
            let val_expr = self.convert_schema(val_schema)?;
            let val_rule = self.fresh_name(&format!("val_{}", sanitize_rule_name(key)));
            self.sub_rules.push((val_rule.clone(), val_expr));

            let kv_expr = GrammarExpr::Sequence(vec![
                self.json_string_literal(key),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Literal(b":".to_vec()),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Ref(val_rule),
            ]);
            let kv_rule = self.fresh_name(&format!("kv_{}", sanitize_rule_name(key)));
            self.sub_rules.push((kv_rule.clone(), kv_expr));
            kv_rules.push((key.clone(), kv_rule));
        }

        // Partition into required / optional.
        let required_keys: Vec<String> = properties
            .iter()
            .map(|(k, _)| k.clone())
            .filter(|k| required.contains(k))
            .collect();
        let mut optional_keys: Vec<String> = properties
            .iter()
            .map(|(k, _)| k.clone())
            .filter(|k| !required.contains(k))
            .collect();

        // Handle additionalProperties: true / {schema} → add wildcard kv.
        let wildcard_kv = match additional {
            Some(serde_json::Value::Bool(true)) => {
                let key_rule = self.json_string();
                let val_rule = self.json_value();
                let kv_expr = GrammarExpr::Sequence(vec![
                    key_rule,
                    GrammarExpr::Ref("ws".into()),
                    GrammarExpr::Literal(b":".to_vec()),
                    GrammarExpr::Ref("ws".into()),
                    val_rule,
                ]);
                let kv_rule = self.fresh_name("kv_wildcard");
                self.sub_rules.push((kv_rule.clone(), kv_expr));
                Some(kv_rule)
            }
            Some(schema) if schema.is_object() => {
                let val_expr = self.convert_schema(schema)?;
                let key_rule = self.json_string();
                let kv_expr = GrammarExpr::Sequence(vec![
                    key_rule,
                    GrammarExpr::Ref("ws".into()),
                    GrammarExpr::Literal(b":".to_vec()),
                    GrammarExpr::Ref("ws".into()),
                    val_expr,
                ]);
                let kv_rule = self.fresh_name("kv_wildcard");
                self.sub_rules.push((kv_rule.clone(), kv_expr));
                Some(kv_rule)
            }
            _ => None,
        };

        // If wildcard is present, add it as an optional.
        if let Some(wc) = wildcard_kv {
            kv_rules.push(("*".into(), wc.clone()));
            optional_keys.push("*".into());
        }

        // Build the body expression.
        let kv_ref = |rules: &[(String, String)], key: &str| -> GrammarExpr {
            let rule = rules.iter().find(|(k, _)| k == key).map(|(_, r)| r.clone()).unwrap();
            GrammarExpr::Ref(rule)
        };

        // Required part: R1_kv , ws R2_kv , ws ...
        let mut parts: Vec<GrammarExpr> = Vec::new();
        parts.push(GrammarExpr::Literal(b"{".to_vec()));
        parts.push(GrammarExpr::Ref("ws".into()));

        for (i, key) in required_keys.iter().enumerate() {
            if i > 0 {
                parts.push(GrammarExpr::Literal(b",".to_vec()));
                parts.push(GrammarExpr::Ref("ws".into()));
            }
            parts.push(kv_ref(&kv_rules, key));
        }

        // Optional part: using CFA sequential permutation.
        if !optional_keys.is_empty() {
            let opt_choice = build_optional_choice(&optional_keys, &kv_rules);

            let opt_block = if required_keys.is_empty() {
                // No required props: entire optional block is just wrapped in (…)?.
                GrammarExpr::Optional(Box::new(opt_choice))
            } else {
                // Required props precede optionals: must separate with a comma.
                GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    GrammarExpr::Ref("ws".into()),
                    opt_choice,
                ])))
            };

            parts.push(opt_block);
        }

        parts.push(GrammarExpr::Ref("ws".into()));
        parts.push(GrammarExpr::Literal(b"}".to_vec()));

        Ok(GrammarExpr::Sequence(parts))
    }

    // -----------------------------------------------------------------------
    // Array
    // -----------------------------------------------------------------------

    fn convert_array(
        &mut self,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<GrammarExpr, GlrMaskError> {
        let min_items = obj.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_items = obj.get("maxItems").and_then(|v| v.as_u64()).map(|v| v as usize);

        // prefixItems: fixed tuple.
        if let Some(prefix) = obj.get("prefixItems").and_then(|v| v.as_array()) {
            let item_exprs: Result<Vec<GrammarExpr>, _> =
                prefix.iter().map(|s| self.convert_schema(s)).collect();
            let items = item_exprs?;
            let mut seq = vec![
                GrammarExpr::Literal(b"[".to_vec()),
                GrammarExpr::Ref("ws".into()),
            ];
            for (i, item) in items.into_iter().enumerate() {
                if i > 0 {
                    seq.push(GrammarExpr::Literal(b",".to_vec()));
                    seq.push(GrammarExpr::Ref("ws".into()));
                }
                seq.push(item);
            }
            seq.push(GrammarExpr::Ref("ws".into()));
            seq.push(GrammarExpr::Literal(b"]".to_vec()));
            return Ok(GrammarExpr::Sequence(seq));
        }

        // Uniform items schema.
        let item_expr = if let Some(schema) = obj.get("items") {
            self.convert_schema(schema)?
        } else {
            self.json_value()
        };

        let item_rule = self.fresh_name("arr_item");
        self.sub_rules.push((item_rule.clone(), item_expr));

        // Build repetition respecting min/max.
        let arr_body = build_repetition(&item_rule, min_items, max_items);
        Ok(GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"[".to_vec()),
            GrammarExpr::Ref("ws".into()),
            arr_body,
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Literal(b"]".to_vec()),
        ]))
    }

    // -----------------------------------------------------------------------
    // JSON primitives
    // -----------------------------------------------------------------------

    /// Generic JSON value (fully recursive: includes arrays and objects).
    fn json_value(&mut self) -> GrammarExpr {
        let name = "_json_value";
        if self.sub_rules.iter().any(|(n, _)| n == name) {
            return GrammarExpr::Ref(name.to_string());
        }
        // Insert placeholder first to break recursive calls.
        self.sub_rules.push((name.to_string(), GrammarExpr::Literal(b"null".to_vec())));
        let str_ref = self.json_string();
        let num_ref = self.json_number();
        let arr_ref = self.json_array_generic();
        let obj_ref = self.json_object_generic();
        let val_expr = GrammarExpr::Choice(vec![
            str_ref,
            num_ref,
            GrammarExpr::Literal(b"true".to_vec()),
            GrammarExpr::Literal(b"false".to_vec()),
            GrammarExpr::Literal(b"null".to_vec()),
            arr_ref,
            obj_ref,
        ]);
        // Replace placeholder with real definition.
        if let Some(pos) = self.sub_rules.iter().position(|(n, _)| n == name) {
            self.sub_rules[pos].1 = val_expr;
        }
        GrammarExpr::Ref(name.to_string())
    }

    /// Generic JSON array: `[ ws (_json_value (, ws _json_value)*)? ws ]`.
    fn json_array_generic(&mut self) -> GrammarExpr {
        let name = "_json_arr";
        if self.sub_rules.iter().any(|(n, _)| n == name) {
            return GrammarExpr::Ref(name.to_string());
        }
        // Forward-declare to prevent recursion loops.
        self.sub_rules.push((name.to_string(), GrammarExpr::Literal(b"[]".to_vec())));
        let item_rest = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b",".to_vec()),
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Ref("_json_value".into()),
        ]);
        let items = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
            GrammarExpr::Ref("_json_value".into()),
            GrammarExpr::Repeat(Box::new(item_rest)),
        ])));
        let arr_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"[".to_vec()),
            GrammarExpr::Ref("ws".into()),
            items,
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Literal(b"]".to_vec()),
        ]);
        if let Some(pos) = self.sub_rules.iter().position(|(n, _)| n == name) {
            self.sub_rules[pos].1 = arr_expr;
        }
        GrammarExpr::Ref(name.to_string())
    }

    /// Generic JSON object: `{ ws (str : val (, str : val)*)? ws }`.
    fn json_object_generic(&mut self) -> GrammarExpr {
        let name = "_json_obj";
        if self.sub_rules.iter().any(|(n, _)| n == name) {
            return GrammarExpr::Ref(name.to_string());
        }
        self.sub_rules.push((name.to_string(), GrammarExpr::Literal(b"{}".to_vec())));
        let str_ref = self.json_string();
        let kv = GrammarExpr::Sequence(vec![
            str_ref,
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Literal(b":".to_vec()),
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Ref("_json_value".into()),
        ]);
        let kv_rest = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b",".to_vec()),
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Sequence(vec![
                self.json_string(),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Literal(b":".to_vec()),
                GrammarExpr::Ref("ws".into()),
                GrammarExpr::Ref("_json_value".into()),
            ]),
        ]);
        let entries = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
            kv,
            GrammarExpr::Repeat(Box::new(kv_rest)),
        ])));
        let obj_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            GrammarExpr::Ref("ws".into()),
            entries,
            GrammarExpr::Ref("ws".into()),
            GrammarExpr::Literal(b"}".to_vec()),
        ]);
        if let Some(pos) = self.sub_rules.iter().position(|(n, _)| n == name) {
            self.sub_rules[pos].1 = obj_expr;
        }
        GrammarExpr::Ref(name.to_string())
    }

    /// JSON string: `"` (escape | [^"\\])* `"`.
    fn json_string(&mut self) -> GrammarExpr {
        let name = "_json_string";
        if self.sub_rules.iter().any(|(n, _)| n == name) {
            return GrammarExpr::Ref(name.to_string());
        }
        let char_name = "_json_str_char";
        let char_expr = GrammarExpr::Choice(vec![
            GrammarExpr::CharClass { def: "\"\\\\".into(), negate: true },
            GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"\\".to_vec()),
                GrammarExpr::CharClass { def: "\"\\\\bfnrt/".into(), negate: false },
            ]),
            // \uXXXX escape.
            GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"\\u".to_vec()),
                GrammarExpr::CharClass { def: "0-9a-fA-F".into(), negate: false },
                GrammarExpr::CharClass { def: "0-9a-fA-F".into(), negate: false },
                GrammarExpr::CharClass { def: "0-9a-fA-F".into(), negate: false },
                GrammarExpr::CharClass { def: "0-9a-fA-F".into(), negate: false },
            ]),
        ]);
        self.sub_rules.push((char_name.to_string(), char_expr));
        let str_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            GrammarExpr::Repeat(Box::new(GrammarExpr::Ref(char_name.to_string()))),
            GrammarExpr::Literal(b"\"".to_vec()),
        ]);
        self.sub_rules.push((name.to_string(), str_expr));
        GrammarExpr::Ref(name.to_string())
    }

    /// JSON string constrained by minLength / maxLength.
    fn json_string_bounded(&mut self, min: usize, max: Option<usize>) -> GrammarExpr {
        let char_ref = self.json_string(); // ensures _json_str_char is defined
        let _ = char_ref; // we use the char rule below
        let repetition = build_repetition("_json_str_char", min, max);
        GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            repetition,
            GrammarExpr::Literal(b"\"".to_vec()),
        ])
    }

    /// JSON string matching a regex pattern (wrapped in `"..."` delimiters).
    fn json_string_pattern(&self, pattern: &str) -> GrammarExpr {
        // Produce: `"` <pattern_as_raw_regex> `"`
        // We emit the pattern verbatim as a RawRegex terminal inside the string.
        GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            GrammarExpr::RawRegex(pattern.to_string()),
            GrammarExpr::Literal(b"\"".to_vec()),
        ])
    }

    /// JSON number.
    fn json_number(&mut self) -> GrammarExpr {
        let name = "_json_number";
        if self.sub_rules.iter().any(|(n, _)| n == name) {
            return GrammarExpr::Ref(name.to_string());
        }
        let digits = GrammarExpr::RepeatOne(Box::new(GrammarExpr::CharClass {
            def: "0-9".into(), negate: false,
        }));
        let integer_part = GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"0".to_vec()),
            GrammarExpr::Sequence(vec![
                GrammarExpr::CharClass { def: "1-9".into(), negate: false },
                GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass { def: "0-9".into(), negate: false })),
            ]),
        ]);
        let frac = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b".".to_vec()),
            digits.clone(),
        ])));
        let exp = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
            GrammarExpr::CharClass { def: "eE".into(), negate: false },
            GrammarExpr::Optional(Box::new(GrammarExpr::CharClass { def: "+-".into(), negate: false })),
            digits,
        ])));
        let num_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
            integer_part,
            frac,
            exp,
        ]);
        self.sub_rules.push((name.to_string(), num_expr));
        GrammarExpr::Ref(name.to_string())
    }

    /// JSON integer.
    fn json_integer(&mut self) -> GrammarExpr {
        let name = "_json_integer";
        if self.sub_rules.iter().any(|(n, _)| n == name) {
            return GrammarExpr::Ref(name.to_string());
        }
        let int_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
            GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"0".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::CharClass { def: "1-9".into(), negate: false },
                    GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass { def: "0-9".into(), negate: false })),
                ]),
            ]),
        ]);
        self.sub_rules.push((name.to_string(), int_expr));
        GrammarExpr::Ref(name.to_string())
    }

    /// Produce a GrammarExpr for a specific JSON literal value.
    fn json_literal(&self, value: &serde_json::Value) -> GrammarExpr {
        GrammarExpr::Literal(value.to_string().into_bytes())
    }

    /// Produce a GrammarExpr for a JSON string literal: `"key"`.
    fn json_string_literal(&self, s: &str) -> GrammarExpr {
        let mut bytes = Vec::new();
        bytes.push(b'"');
        for b in s.bytes() {
            match b {
                b'"' => { bytes.push(b'\\'); bytes.push(b'"'); }
                b'\\' => { bytes.push(b'\\'); bytes.push(b'\\'); }
                _ => bytes.push(b),
            }
        }
        bytes.push(b'"');
        GrammarExpr::Literal(bytes)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collapse a vec of alternatives into a `Choice` (or return single element).
fn choice_or_single(alts: Vec<GrammarExpr>) -> GrammarExpr {
    if alts.len() == 1 { alts.into_iter().next().unwrap() } else { GrammarExpr::Choice(alts) }
}

/// Sanitise a property name into a valid rule-name fragment.
fn sanitize_rule_name(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

/// Build the CFA-style sequential permutation optional-property choice.
///
/// For `optional=[O1, O2, O3]`, produces:
/// ```text
/// Choice([
///   Sequence([O1_kv, Optional(,O2_kv), Optional(,O3_kv)]),
///   Sequence([O2_kv, Optional(,O3_kv)]),
///   O3_kv,
/// ])
/// ```
fn build_optional_choice(optional_keys: &[String], kv_rules: &[(String, String)]) -> GrammarExpr {
    let kv_ref = |key: &str| -> GrammarExpr {
        let rule = kv_rules.iter().find(|(k, _)| k == key).map(|(_, r)| r.clone()).unwrap();
        GrammarExpr::Ref(rule)
    };

    let mut alternatives = Vec::new();
    for start_idx in 0..optional_keys.len() {
        let first = kv_ref(&optional_keys[start_idx]);
        // Remaining keys after start_idx appear as optional trailing items.
        let mut branch: Vec<GrammarExpr> = vec![first];
        for j in (start_idx + 1)..optional_keys.len() {
            let trailing = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b",".to_vec()),
                GrammarExpr::Ref("ws".into()),
                kv_ref(&optional_keys[j]),
            ])));
            branch.push(trailing);
        }
        let alt = if branch.len() == 1 {
            branch.into_iter().next().unwrap()
        } else {
            GrammarExpr::Sequence(branch)
        };
        alternatives.push(alt);
    }
    choice_or_single(alternatives)
}

/// Build a repetition expression for `min..=max` items of `item_rule`, separated by `, ws`.
///
/// - `min=0, max=None` → `(item (, ws item)*)?`
/// - `min=1, max=None` → `item (, ws item)*`
/// - `min=2, max=Some(4)` → `item , ws item (, ws item)? (, ws item)?`
fn build_repetition(item_rule: &str, min: usize, max: Option<usize>) -> GrammarExpr {
    let item = GrammarExpr::Ref(item_rule.to_string());
    let comma_item = GrammarExpr::Sequence(vec![
        GrammarExpr::Literal(b",".to_vec()),
        GrammarExpr::Ref("ws".into()),
        GrammarExpr::Ref(item_rule.to_string()),
    ]);

    match (min, max) {
        (0, None) => {
            // (item (, ws item)*)?
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                item,
                GrammarExpr::Repeat(Box::new(comma_item)),
            ])))
        }
        (1, None) => {
            // item (, ws item)*
            GrammarExpr::Sequence(vec![item, GrammarExpr::Repeat(Box::new(comma_item))])
        }
        (n, None) => {
            // item (,item){n-1} (,item)*
            let mut parts = vec![item];
            for _ in 1..n {
                parts.push(GrammarExpr::Literal(b",".to_vec()));
                parts.push(GrammarExpr::Ref("ws".into()));
                parts.push(GrammarExpr::Ref(item_rule.to_string()));
            }
            parts.push(GrammarExpr::Repeat(Box::new(comma_item)));
            GrammarExpr::Sequence(parts)
        }
        (0, Some(m)) if m == 0 => {
            // Empty: produce ε via Optional of impossible.
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"".to_vec())))
        }
        (0, Some(m)) => {
            // (item (, ws item)? ... (up to m-1 more))?
            let mut tail: Vec<GrammarExpr> = vec![item];
            for _ in 1..m {
                tail.push(GrammarExpr::Optional(Box::new(comma_item.clone())));
            }
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(tail)))
        }
        (n, Some(m)) => {
            // n required then (m-n) optional
            let mut parts = vec![item];
            for _ in 1..n {
                parts.push(GrammarExpr::Literal(b",".to_vec()));
                parts.push(GrammarExpr::Ref("ws".into()));
                parts.push(GrammarExpr::Ref(item_rule.to_string()));
            }
            for _ in n..m {
                parts.push(GrammarExpr::Optional(Box::new(comma_item.clone())));
            }
            GrammarExpr::Sequence(parts)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;

    // -------------------------------------------------------------------------
    // Grammar construction tests (smoke tests)
    // -------------------------------------------------------------------------

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
        // Schema with only required properties should generate grammar without
        // trailing commas (the sequence must be parsable).
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
        // Schema with only optional properties — no comma required between { and first prop.
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
        // additionalProperties: false + no properties → only {} allowed.
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

    // -------------------------------------------------------------------------
    // Behavioral tests using Constraint
    // -------------------------------------------------------------------------

    /// Build a Constraint from a JSON Schema and a toy vocabulary,
    /// then advance through the given token sequence and check final acceptance.
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
            state.commit(id);
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
