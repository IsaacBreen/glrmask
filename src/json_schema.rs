//! JSON Schema to Grammar conversion.
//!
//! This module converts JSON Schema (draft-07 compatible) to Sep1's grammar representation.
//! The goal is to generate grammars that are as permissive as the schema allows, without
//! trying to enforce semantic constraints that are impossible for CFGs.
//!
//! # Supported features:
//! - `type`: object, array, string, integer, number, boolean, null, and arrays of types
//! - `properties`, `additionalProperties`
//! - `items`, `prefixItems` (draft 2020-12)
//! - `$ref`, `$defs`, `definitions`
//! - `allOf`, `anyOf`, `oneOf`
//! - `const`, `enum`
//!
//! # Unsupported (intentionally - these require semantic validation, not syntax):
//! - minimum/maximum/exclusiveMinimum/exclusiveMaximum
//! - minLength/maxLength
//! - minItems/maxItems
//! - minProperties/maxProperties
//! - pattern, format, uniqueItems, dependencies, if/then/else, not

use crate::interface::GrammarExpr;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// JSON Schema to Grammar converter.
pub struct JsonSchemaConverter {
    root_schema: Value,
    rule_counter: usize,
    /// Maps $ref paths to rule names
    resolved_refs: BTreeMap<String, String>,
    /// Set of refs that have been converted to rules
    generated_refs: HashSet<String>,
    /// Stack of refs currently being processed (for detecting self-references)
    current_ref_stack: Vec<String>,
    /// Stack of refs currently being resolved (for detecting cycles during inlining analysis)
    resolving_stack: Vec<String>,
    /// Cache of inlined references: path -> expr
    inlined_refs: BTreeMap<String, GrammarExpr>,
    /// Map of definition paths to their schemas
    definitions: BTreeMap<String, Value>,
    /// Rules to be emitted: (name, expr)
    rules: Vec<(String, GrammarExpr)>,
    /// Deduplication map: Expression -> Rule Name
    rule_dedup: BTreeMap<GrammarExpr, String>,
    /// Alias map: Duplicate Rule Name -> Original Rule Name
    aliases: BTreeMap<String, String>,
    /// Queue of pending refs to process: (ref_path, schema)
    pending_refs: Vec<(String, Value)>,
    /// Track which primitive rules are needed
    needs_json_value: bool,
    needs_json_object: bool,
    needs_json_array: bool,
    needs_json_kv: bool,
}

impl JsonSchemaConverter {
    /// Create a new converter from a JSON schema value.
    pub fn new(schema: Value) -> Self {
        Self {
            root_schema: schema,
            rule_counter: 0,
            resolved_refs: BTreeMap::new(),
            generated_refs: HashSet::new(),
            current_ref_stack: Vec::new(),
            resolving_stack: Vec::new(),
            inlined_refs: BTreeMap::new(),
            definitions: BTreeMap::new(),
            rules: Vec::new(),
            rule_dedup: BTreeMap::new(),
            aliases: BTreeMap::new(),
            pending_refs: Vec::new(),
            needs_json_value: false,
            needs_json_object: false,
            needs_json_array: false,
            needs_json_kv: false,
        }
    }

    /// Generate a new unique rule name with the given prefix.
    fn new_rule_name(&mut self, prefix: &str) -> String {
        self.rule_counter += 1;
        format!("_{}{}", prefix, self.rule_counter)
    }

    /// Add a rule to the grammar.
    fn add_rule(&mut self, name: String, expr: GrammarExpr) {
        if let Some(existing_name) = self.rule_dedup.get(&expr) {
            if *existing_name != name {
                self.aliases.insert(name, existing_name.clone());
            }
            return;
        }
        self.rule_dedup.insert(expr.clone(), name.clone());
        self.rules.push((name, expr));
    }
    
    fn apply_aliases(&mut self) {
        // Resolve alias chains
        let mut resolved_aliases = self.aliases.clone();
        let keys: Vec<String> = self.aliases.keys().cloned().collect();
        // Simple loop to resolve chains (safe upper bound)
        for _ in 0..keys.len() { 
             let mut changed = false;
             for val in resolved_aliases.values_mut() {
                 if let Some(target) = self.aliases.get(val) {
                     *val = target.clone();
                     changed = true;
                 }
             }
             if !changed { break; }
        }

        // Apply to all rules
        for (_, expr) in &mut self.rules {
             Self::update_expr_refs(expr, &resolved_aliases);
        }
    }

    fn update_expr_refs(expr: &mut GrammarExpr, aliases: &BTreeMap<String, String>) {
        match expr {
            GrammarExpr::Ref(name) => {
                if let Some(target) = aliases.get(name) {
                    *name = target.clone();
                }
            },
            GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                for e in exprs {
                    Self::update_expr_refs(e, aliases);
                }
            },
            GrammarExpr::Optional(e) | GrammarExpr::Repeat(e) => {
                Self::update_expr_refs(e, aliases);
            },
            _ => {}
        }
    }

    /// Reference _json_value and mark it as needed
    fn json_value_ref(&mut self) -> GrammarExpr {
        self.needs_json_value = true;
        GrammarExpr::Ref("_json_value".to_string())
    }
    
    /// Reference _json_object and mark it as needed
    fn json_object_ref(&mut self) -> GrammarExpr {
        self.needs_json_object = true;
        GrammarExpr::Ref("_json_object".to_string())
    }
    
    /// Reference _json_array and mark it as needed
    fn json_array_ref(&mut self) -> GrammarExpr {
        self.needs_json_array = true;
        GrammarExpr::Ref("_json_array".to_string())
    }
    
    /// Reference _json_kv and mark it as needed
    fn json_kv_ref(&mut self) -> GrammarExpr {
        self.needs_json_kv = true;
        GrammarExpr::Ref("_json_kv".to_string())
    }
    /// Convert the schema to a list of grammar rules.
    /// Returns (rules, root_rule_name).
    pub fn convert(mut self) -> Result<(Vec<(String, GrammarExpr)>, String), String> {
        // Register all definitions first
        self.register_definitions();

        // Generate main rule from root schema
        let root_rule = "root".to_string();
        self.convert_schema(&self.root_schema.clone(), root_rule.clone())?;

        // Process pending refs
        while !self.pending_refs.is_empty() {
            let (ref_path, def_schema) = self.pending_refs.remove(0);
            if !self.generated_refs.contains(&ref_path) {
                self.generated_refs.insert(ref_path.clone());
                let rule_name = self.resolved_refs.get(&ref_path).cloned()
                    .unwrap_or_else(|| {
                        let name = self.new_rule_name("def");
                        self.resolved_refs.insert(ref_path.clone(), name.clone());
                        name
                    });
                self.current_ref_stack.push(ref_path.clone());
                self.convert_schema(&def_schema, rule_name)?;
                self.current_ref_stack.pop();
            }
        }

        // Add primitive rules
        self.add_primitive_rules();
        
        // Resolve aliases
        self.apply_aliases();

        Ok((self.rules, root_rule))
    }

    /// Pre-register all $defs/definitions for forward references.
    fn register_definitions(&mut self) {
        let keys = ["$defs", "definitions"];
        for key in &keys {
            if let Some(defs) = self.root_schema.get(*key).and_then(|v| v.as_object()).cloned() {
                for (name, def_schema) in defs {
                    let ref_path = format!("#/{}/{}", key, name);
                    self.definitions.insert(ref_path, def_schema);
                }
            }
        }
    }

    /// Resolve a $ref and return an expression (inlined or ref).
    fn resolve_ref_expr(&mut self, ref_path: &str) -> Option<GrammarExpr> {
        // 1. Check if already inlined
        if let Some(expr) = self.inlined_refs.get(ref_path) {
            return Some(expr.clone());
        }
        
        // 2. Check if already resolved to a rule
        if let Some(name) = self.resolved_refs.get(ref_path) {
            return Some(GrammarExpr::Ref(name.clone()));
        }
        
        // 3. Cycle detection
        if self.resolving_stack.contains(&ref_path.to_string()) {
            // Recursive reference detected during inlining analysis.
            // We must use a named rule. Assign one now.
            let rule_name = self.new_rule_name("rec");
            self.resolved_refs.insert(ref_path.to_string(), rule_name.clone());
            // Schema will be processed when popping stack or explicitly pushed
            return Some(GrammarExpr::Ref(rule_name));
        }

        // 4. Find the schema
        let target_schema = if let Some(schema) = self.definitions.get(ref_path) {
            schema.clone()
        } else if ref_path.starts_with("#/") {
            let parts: Vec<&str> = ref_path[2..].split('/').collect();
            let mut target = self.root_schema.clone();
            for part in &parts {
                target = match target.get(*part) {
                    Some(t) => t.clone(),
                    None => return None,
                };
            }
            target
        } else {
            return None;
        };

        // 5. Try to inline
        self.resolving_stack.push(ref_path.to_string());
        
        let result = match self.convert_schema_inline(&target_schema) {
            Ok(expr) => {
                // If a rule name was assigned during recursion (step 3 hit), we MUST make it a rule
                if let Some(rule_name) = self.resolved_refs.get(ref_path) {
                     self.pending_refs.push((ref_path.to_string(), target_schema));
                     GrammarExpr::Ref(rule_name.clone())
                } else {
                     // Success! Cache inline.
                     self.inlined_refs.insert(ref_path.to_string(), expr.clone());
                     expr
                }
            },
            Err(_) => {
                // Complex schema, needs a rule.
                // Check if one was already assigned
                if !self.resolved_refs.contains_key(ref_path) {
                    let new_name = self.new_rule_name("def");
                    self.resolved_refs.insert(ref_path.to_string(), new_name);
                }
                let rule_name = self.resolved_refs.get(ref_path).unwrap().clone();
                
                self.pending_refs.push((ref_path.to_string(), target_schema));
                GrammarExpr::Ref(rule_name)
            }
        };
        
        self.resolving_stack.pop();
        Some(result)
    }

    /// Try to convert a schema to an inline grammar expression without creating a named rule.
    /// Returns Ok(expr) for simple schemas that can be inlined, Err(_) for complex schemas
    /// that need their own named rule.
    fn convert_schema_inline(&mut self, schema: &Value) -> Result<GrammarExpr, String> {
        // Handle boolean schemas
        if let Some(b) = schema.as_bool() {
            return if b { 
                Ok(self.json_value_ref()) 
            } else { 
                Ok(GrammarExpr::Literal(b"<NEVER>".to_vec())) 
            };
        }
        
        let obj = schema.as_object().ok_or("complex")?;
        
        // Handle $ref - return expression (inlined or ref)
        if let Some(ref_val) = obj.get("$ref").and_then(|v| v.as_str()) {
            if let Some(expr) = self.resolve_ref_expr(ref_val) {
                return Ok(expr);
            }
            return Ok(self.json_value_ref());
        }
        
        // Handle const
        if let Some(const_val) = obj.get("const") {
            return Ok(self.value_to_literal(const_val));
        }
        
        // Handle enum
        if let Some(enum_vals) = obj.get("enum").and_then(|v| v.as_array()) {
            let alternatives: Vec<GrammarExpr> = enum_vals.iter()
                .map(|v| self.value_to_literal(v))
                .collect();
            if alternatives.len() == 1 {
                return Ok(alternatives[0].clone());
            }
            return Ok(GrammarExpr::Choice(alternatives));
        }
        
        // Handle allOf - try to merge and inline
        if let Some(all_of) = obj.get("allOf").and_then(|v| v.as_array()) {
            // Try to merge allOf. If successful, see if the result is inlineable.
            if let Ok(merged) = self.merge_all_of(all_of, obj) {
                if let Ok(expr) = self.convert_schema_inline(&merged) {
                    return Ok(expr);
                }
            }
            // If merge fails or result is complex, fall through to Err("complex")
            // But wait, if we don't handle it here, it drops to "complex" error at end.
            // Which forces a rule. Correct.
        }
        
        // Handle simple types (no additional constraints that matter)
        if let Some(type_val) = obj.get("type") {
            if let Some(type_str) = type_val.as_str() {
                return match type_str {
                    "string" => Ok(GrammarExpr::Ref("JSON_STRING".to_string())),
                    "integer" => Ok(GrammarExpr::Ref("JSON_INTEGER".to_string())),
                    "number" => Ok(GrammarExpr::Ref("JSON_NUMBER".to_string())),
                    "boolean" => Ok(GrammarExpr::Ref("JSON_BOOL".to_string())),
                    "null" => Ok(GrammarExpr::Ref("JSON_NULL".to_string())),
                    // Object without properties -> generic object
                    "object" => {
                        let has_properties = obj.get("properties").and_then(|v| v.as_object()).map(|p| !p.is_empty()).unwrap_or(false);
                        if !has_properties && obj.get("additionalProperties") != Some(&Value::Bool(false)) {
                            Ok(self.json_object_ref())
                        } else {
                            Err("complex".to_string())
                        }
                    }
                    // Array without items -> generic array
                    "array" => {
                        if obj.get("items").is_none() && obj.get("prefixItems").is_none() {
                            Ok(self.json_array_ref())
                        } else {
                            Err("complex".to_string())
                        }
                    }
                    _ => Ok(self.json_value_ref()),
                };
            } else if let Some(types) = type_val.as_array() {
                // Multi-type: inline if all are primitives OR simple object/array
                let mut alternatives = Vec::new();
                let mut all_inline = true;
                for t in types {
                    if let Some(type_str) = t.as_str() {
                        match type_str {
                            "string" => alternatives.push(GrammarExpr::Ref("JSON_STRING".to_string())),
                            "integer" => alternatives.push(GrammarExpr::Ref("JSON_INTEGER".to_string())),
                            "number" => alternatives.push(GrammarExpr::Ref("JSON_NUMBER".to_string())),
                            "boolean" => alternatives.push(GrammarExpr::Ref("JSON_BOOL".to_string())),
                            "null" => alternatives.push(GrammarExpr::Ref("JSON_NULL".to_string())),
                            "object" => {
                                let has_props = obj.get("properties").and_then(|v| v.as_object()).map(|p| !p.is_empty()).unwrap_or(false);
                                if !has_props && obj.get("additionalProperties") != Some(&Value::Bool(false)) {
                                    alternatives.push(self.json_object_ref());
                                } else { all_inline = false; break; }
                            }
                            "array" => {
                                if obj.get("items").is_none() && obj.get("prefixItems").is_none() {
                                    alternatives.push(self.json_array_ref());
                                } else { all_inline = false; break; }
                            }
                            _ => { all_inline = false; break; }
                        }
                    } else {
                        all_inline = false;
                        break;
                    }
                }
                if all_inline && !alternatives.is_empty() {
                    return Ok(GrammarExpr::Choice(alternatives));
                }
            }
        }
        
        // For complex schemas (allOf, anyOf, oneOf, objects, arrays), need a named rule
        
        // Handle anyOf / oneOf - inline if all alternatives are inlineable
        if let Some(any_of) = obj.get("anyOf").or_else(|| obj.get("oneOf")).and_then(|v| v.as_array()) {
            // Check if parent has merging properties - if so, it's complex
            // Note: We ignore "type" here as it's often consistent with alternatives
            let has_parent_props = obj.contains_key("properties") || 
                                   obj.contains_key("additionalProperties");
            
            if !has_parent_props {
                let mut alternatives = Vec::new();
                let mut all_inline = true;
                
                for sub in any_of {
                    if let Ok(expr) = self.convert_schema_inline(sub) {
                        alternatives.push(expr);
                    } else {
                        all_inline = false;
                        break;
                    }
                }
                
                if all_inline && !alternatives.is_empty() {
                    return Ok(GrammarExpr::Choice(alternatives));
                }
            }
        }

        // If no type/combining keywords, it's a generic JSON value
        if !obj.contains_key("type") && !obj.contains_key("allOf") && 
           !obj.contains_key("anyOf") && !obj.contains_key("oneOf") &&
           !obj.contains_key("properties") && !obj.contains_key("items") {
             return Ok(self.json_value_ref());
        }

        Err("complex".to_string())
    }

    /// Convert a schema to grammar rules. Returns the rule name.
    fn convert_schema(&mut self, schema: &Value, rule_name: String) -> Result<String, String> {
        // Handle boolean schemas
        if let Some(b) = schema.as_bool() {
            if b {
                let ref_expr = self.json_value_ref();
                self.add_rule(rule_name.clone(), ref_expr);
            } else {
                // false schema - nothing matches
                self.add_rule(rule_name.clone(), GrammarExpr::Literal(b"<NEVER>".to_vec()));
            }
            return Ok(rule_name);
        }

        let obj = schema.as_object().ok_or("Invalid schema: expected object or boolean")?;

        // Handle $ref
        if let Some(ref_val) = obj.get("$ref").and_then(|v| v.as_str()) {
            if let Some(expr) = self.resolve_ref_expr(ref_val) {
                self.add_rule(rule_name.clone(), expr);
                return Ok(rule_name);
            } else {
                let ref_expr = self.json_value_ref();
                self.add_rule(rule_name.clone(), ref_expr);
                return Ok(rule_name);
            }
        }

        // Handle allOf
        if let Some(all_of) = obj.get("allOf").and_then(|v| v.as_array()) {
            let merged = self.merge_all_of(all_of, obj)?;
            return self.convert_schema(&merged, rule_name);
        }

        // Handle anyOf / oneOf
        // For anyOf, we need to merge parent properties into each alternative.
        // This handles cases like PackageJson where the top-level has properties
        // but also has anyOf for additional constraints.
        if let Some(any_of) = obj.get("anyOf").or_else(|| obj.get("oneOf")).and_then(|v| v.as_array()) {
            // Check if parent has properties or type that should be merged
            let has_parent_props = obj.contains_key("properties") || 
                                   obj.contains_key("type") ||
                                   obj.contains_key("additionalProperties");
            
            let mut alternatives = Vec::new();
            for sub in any_of {
                if has_parent_props {
                    // Merge parent properties into this alternative - needs named rule
                    let sub_name = self.new_rule_name("alt");
                    let merged = self.merge_anyof_with_parent(sub, obj)?;
                    self.convert_schema(&merged, sub_name.clone())?;
                    alternatives.push(GrammarExpr::Ref(sub_name));
                } else {
                    // Try to inline simple alternatives
                    match self.convert_schema_inline(sub) {
                        Ok(expr) => alternatives.push(expr),
                        Err(_) => {
                            let sub_name = self.new_rule_name("alt");
                            self.convert_schema(sub, sub_name.clone())?;
                            alternatives.push(GrammarExpr::Ref(sub_name));
                        }
                    }
                }
            }
            self.add_rule(rule_name.clone(), GrammarExpr::Choice(alternatives));
            return Ok(rule_name);
        }

        // Handle const
        if let Some(const_val) = obj.get("const") {
            let literal = self.value_to_literal(const_val);
            self.add_rule(rule_name.clone(), literal);
            return Ok(rule_name);
        }

        // Handle enum
        if let Some(enum_vals) = obj.get("enum").and_then(|v| v.as_array()) {
            let alternatives: Vec<GrammarExpr> = enum_vals.iter()
                .map(|v| self.value_to_literal(v))
                .collect();
            self.add_rule(rule_name.clone(), GrammarExpr::Choice(alternatives));
            return Ok(rule_name);
        }

        // Handle type
        let schema_type = obj.get("type");

        if let Some(type_val) = schema_type {
            if let Some(type_str) = type_val.as_str() {
                return self.convert_typed_schema(type_str, obj, rule_name);
            } else if let Some(types) = type_val.as_array() {
                // Multiple types - inline simple primitives
                let mut alternatives = Vec::new();
                for t in types {
                    if let Some(type_str) = t.as_str() {
                        // Inline primitive types directly
                        match type_str {
                            "string" => alternatives.push(GrammarExpr::Ref("JSON_STRING".to_string())),
                            "integer" => alternatives.push(GrammarExpr::Ref("JSON_INTEGER".to_string())),
                            "number" => alternatives.push(GrammarExpr::Ref("JSON_NUMBER".to_string())),
                            "boolean" => alternatives.push(GrammarExpr::Ref("JSON_BOOL".to_string())),
                            "null" => alternatives.push(GrammarExpr::Ref("JSON_NULL".to_string())),
                            "object" | "array" => {
                                // Complex types need their own rules
                                let alt_name = self.new_rule_name("type");
                                self.convert_typed_schema(type_str, obj, alt_name.clone())?;
                                alternatives.push(GrammarExpr::Ref(alt_name));
                            }
                            _ => alternatives.push(self.json_value_ref()),
                        }
                    }
                }
                self.add_rule(rule_name.clone(), GrammarExpr::Choice(alternatives));
                return Ok(rule_name);
            }
        }

        // No type specified - allow any JSON value
        let ref_expr = self.json_value_ref();
        self.add_rule(rule_name.clone(), ref_expr);
        Ok(rule_name)
    }

    /// Convert a schema with a known type.
    fn convert_typed_schema(&mut self, type_str: &str, obj: &serde_json::Map<String, Value>, rule_name: String) -> Result<String, String> {
        match type_str {
            "object" => self.convert_object(obj, rule_name),
            "array" => self.convert_array(obj, rule_name),
            "string" => {
                self.add_rule(rule_name.clone(), GrammarExpr::Ref("JSON_STRING".to_string()));
                Ok(rule_name)
            }
            "integer" => {
                self.add_rule(rule_name.clone(), GrammarExpr::Ref("JSON_INTEGER".to_string()));
                Ok(rule_name)
            }
            "number" => {
                self.add_rule(rule_name.clone(), GrammarExpr::Ref("JSON_NUMBER".to_string()));
                Ok(rule_name)
            }
            "boolean" => {
                self.add_rule(rule_name.clone(), GrammarExpr::Ref("JSON_BOOL".to_string()));
                Ok(rule_name)
            }
            "null" => {
                self.add_rule(rule_name.clone(), GrammarExpr::Ref("JSON_NULL".to_string()));
                Ok(rule_name)
            }
            _ => {
                let ref_expr = self.json_value_ref();
                self.add_rule(rule_name.clone(), ref_expr);
                Ok(rule_name)
            }
        }
    }

    /// Convert an object schema.
    fn convert_object(&mut self, obj: &serde_json::Map<String, Value>, rule_name: String) -> Result<String, String> {
        let properties = obj.get("properties").and_then(|v| v.as_object());
        let additional_props = obj.get("additionalProperties");

        // If no properties defined and additional allowed, just use generic object
        let has_properties = properties.map(|p| !p.is_empty()).unwrap_or(false);
        if !has_properties && additional_props != Some(&Value::Bool(false)) {
            let ref_expr = self.json_object_ref();
            self.add_rule(rule_name.clone(), ref_expr);
            return Ok(rule_name);
        }

        // If no properties and no additional allowed, empty object only
        if !has_properties && additional_props == Some(&Value::Bool(false)) {
            self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"{".to_vec()),
                GrammarExpr::Literal(b"}".to_vec()),
            ]));
            return Ok(rule_name);
        }

        // Build member alternatives
        let mut member_alternatives = Vec::new();

        if let Some(props) = properties {
            for (prop_name, prop_schema) in props {
                // Try to inline simple types, only create named rules for complex schemas
                let prop_value_expr = match self.convert_schema_inline(prop_schema) {
                    Ok(expr) => expr,
                    Err(_) => {
                        // Complex schema - create a named rule
                        let prop_value_rule = self.new_rule_name("pv");
                        self.convert_schema(prop_schema, prop_value_rule.clone())?;
                        GrammarExpr::Ref(prop_value_rule)
                    }
                };

                // Build: '"propName"' ':' value
                // WS is handled by the ignore terminal, no need to include it explicitly
                let escaped_name = self.escape_string_for_json(prop_name);
                member_alternatives.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(format!("\"{}\"", escaped_name).into_bytes()),
                    GrammarExpr::Literal(b":".to_vec()),
                    prop_value_expr,
                ]));
            }
        }

        // If additional properties allowed, add generic kv
        // Note: We treat unspecified additionalProperties as false (stricter interpretation)
        // This avoids adding recursive _json_* rules which can slow down optimization.
        // Only explicitly set additionalProperties: true will enable arbitrary properties.
        match additional_props {
            Some(Value::Bool(true)) => {
                member_alternatives.push(self.json_kv_ref());
            }
            Some(Value::Object(ap_schema)) => {
                // Try to inline simple additionalProperties schemas
                let ap_value = Value::Object(ap_schema.clone());
                let ap_expr = match self.convert_schema_inline(&ap_value) {
                    Ok(expr) => expr,
                    Err(_) => {
                        let additional_rule = self.new_rule_name("ap");
                        self.convert_schema(&ap_value, additional_rule.clone())?;
                        GrammarExpr::Ref(additional_rule)
                    }
                };
                member_alternatives.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref("JSON_STRING".to_string()),
                    GrammarExpr::Literal(b":".to_vec()),
                    ap_expr,
                ]));
            }
            _ => {} // additionalProperties: false - don't add generic kv
        }

        // Create member rule
        let member_rule = self.new_rule_name("mem");
        if member_alternatives.len() == 1 {
            self.add_rule(member_rule.clone(), member_alternatives.remove(0));
        } else {
            self.add_rule(member_rule.clone(), GrammarExpr::Choice(member_alternatives));
        }

        // Object rule: { member (, member)* }
        // Build: '{' ( member ( ',' member )* )? '}'
        // WS is handled by the ignore terminal
        let comma_member = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b",".to_vec()),
            GrammarExpr::Ref(member_rule.clone()),
        ]);

        let members_opt = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
            GrammarExpr::Ref(member_rule),
            GrammarExpr::Repeat(Box::new(comma_member)),
        ])));

        self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            members_opt,
            GrammarExpr::Literal(b"}".to_vec()),
        ]));

        Ok(rule_name)
    }

    /// Convert an array schema.
    fn convert_array(&mut self, obj: &serde_json::Map<String, Value>, rule_name: String) -> Result<String, String> {
        let items = obj.get("items");
        let prefix_items = obj.get("prefixItems");

        if items.is_none() && prefix_items.is_none() {
            let ref_expr = self.json_array_ref();
            self.add_rule(rule_name.clone(), ref_expr);
            return Ok(rule_name);
        }

        if let Some(pi) = prefix_items {
            if let Some(pi_array) = pi.as_array() {
                return self.convert_tuple_array(obj, rule_name, pi_array);
            }
        }

        if let Some(item_schema) = items {
            if let Some(b) = item_schema.as_bool() {
                if b {
                    let ref_expr = self.json_array_ref();
                    self.add_rule(rule_name.clone(), ref_expr);
                } else {
                    // Empty array only
                    self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                        GrammarExpr::Literal(b"[".to_vec()),
                        GrammarExpr::Literal(b"]".to_vec()),
                    ]));
                }
                return Ok(rule_name);
            }

            if let Some(_) = item_schema.as_object() {
                // All items must match schema - try to inline simple types
                let item_expr = match self.convert_schema_inline(item_schema) {
                    Ok(expr) => expr,
                    Err(_) => {
                        let item_rule = self.new_rule_name("item");
                        self.convert_schema(item_schema, item_rule.clone())?;
                        GrammarExpr::Ref(item_rule)
                    }
                };

                // Build: '[' ( item ( ',' item )* )? ']'
                // WS is handled by the ignore terminal
                let comma_item = GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    item_expr.clone(),
                ]);

                let items_opt = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    item_expr,
                    GrammarExpr::Repeat(Box::new(comma_item)),
                ])));

                self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"[".to_vec()),
                    items_opt,
                    GrammarExpr::Literal(b"]".to_vec()),
                ]));
                return Ok(rule_name);
            }

            if let Some(items_arr) = item_schema.as_array() {
                // Tuple-style (draft-07)
                return self.convert_tuple_array(obj, rule_name, items_arr);
            }
        }

        // Fallback
        let ref_expr = self.json_array_ref();
        self.add_rule(rule_name.clone(), ref_expr);
        Ok(rule_name)
    }

    /// Convert tuple-style array.
    fn convert_tuple_array(&mut self, obj: &serde_json::Map<String, Value>, rule_name: String, prefix_items: &[Value]) -> Result<String, String> {
        let additional_items = obj.get("additionalItems")
            .or_else(|| obj.get("items"))
            .cloned()
            .unwrap_or(Value::Bool(true));

        if prefix_items.is_empty() {
            if additional_items.as_bool() == Some(true) || additional_items.is_object() {
                let ref_expr = self.json_array_ref();
                self.add_rule(rule_name.clone(), ref_expr);
            } else {
                self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"[".to_vec()),
                    GrammarExpr::Literal(b"]".to_vec()),
                ]));
            }
            return Ok(rule_name);
        }

        // Generate rules for each prefix item
        let mut item_rules = Vec::new();
        for item_schema in prefix_items {
            let item_rule = self.new_rule_name("ti");
            self.convert_schema(item_schema, item_rule.clone())?;
            item_rules.push(item_rule);
        }

        // Build body: first item, then rest with commas
        // WS is handled by the ignore terminal
        let mut body_parts = vec![GrammarExpr::Ref(item_rules[0].clone())];
        for item_rule in &item_rules[1..] {
            body_parts.push(GrammarExpr::Literal(b",".to_vec()));
            body_parts.push(GrammarExpr::Ref(item_rule.clone()));
        }

        // Add additional items if allowed
        match &additional_items {
            Value::Bool(true) => {
                body_parts.push(GrammarExpr::Repeat(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    self.json_value_ref(),
                ]))));
            }
            Value::Object(ai_schema) => {
                let add_rule = self.new_rule_name("ai");
                self.convert_schema(&Value::Object(ai_schema.clone()), add_rule.clone())?;
                body_parts.push(GrammarExpr::Repeat(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    GrammarExpr::Ref(add_rule),
                ]))));
            }
            _ => {}
        }

        let body = GrammarExpr::Sequence(body_parts);
        self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"[".to_vec()),
            GrammarExpr::Optional(Box::new(body)),
            GrammarExpr::Literal(b"]".to_vec()),
        ]));

        Ok(rule_name)
    }

    /// Merge allOf subschemas.
    fn merge_all_of(&self, subschemas: &[Value], parent: &serde_json::Map<String, Value>) -> Result<Value, String> {
        let mut merged: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut merged_props: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut merged_required: Vec<String> = Vec::new();

        for sub in subschemas {
            if let Some(obj) = sub.as_object() {
                // Skip self-referential $refs
                if let Some(ref_val) = obj.get("$ref") {
                    if obj.len() == 1 {
                        if let Some(ref_str) = ref_val.as_str() {
                            if self.current_ref_stack.contains(&ref_str.to_string()) {
                                continue;
                            }
                        }
                    }
                }

                // Merge properties
                if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
                    for (k, v) in props {
                        merged_props.insert(k.clone(), v.clone());
                    }
                }

                // Merge required
                if let Some(req) = obj.get("required").and_then(|v| v.as_array()) {
                    for r in req {
                        if let Some(s) = r.as_str() {
                            merged_required.push(s.to_string());
                        }
                    }
                }

                // Copy other keys
                for (k, v) in obj {
                    if k != "properties" && k != "required" {
                        merged.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        // Add sibling keys from parent
        for (k, v) in parent {
            if k != "allOf" {
                merged.insert(k.clone(), v.clone());
            }
        }

        if !merged_props.is_empty() {
            merged.insert("properties".to_string(), Value::Object(merged_props));
        }
        if !merged_required.is_empty() {
            let unique: Vec<Value> = merged_required.into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .map(Value::String)
                .collect();
            merged.insert("required".to_string(), Value::Array(unique));
        }

        Ok(Value::Object(merged))
    }

    /// Merge an anyOf/oneOf subschema with parent properties.
    /// This handles cases like PackageJson where the parent has properties/type
    /// but also has anyOf for additional constraints.
    fn merge_anyof_with_parent(&self, subschema: &Value, parent: &serde_json::Map<String, Value>) -> Result<Value, String> {
        let mut merged: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut merged_props: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut merged_required: Vec<String> = Vec::new();

        // First, add parent properties (type, properties, additionalProperties, etc.)
        for (k, v) in parent {
            // Skip anyOf/oneOf - we're merging into it
            if k == "anyOf" || k == "oneOf" {
                continue;
            }
            // Skip meta keys
            if k == "$schema" || k == "$id" || k == "title" || k == "description" {
                continue;
            }
            if k == "properties" {
                if let Some(props) = v.as_object() {
                    for (pk, pv) in props {
                        merged_props.insert(pk.clone(), pv.clone());
                    }
                }
            } else if k == "required" {
                if let Some(req) = v.as_array() {
                    for r in req {
                        if let Some(s) = r.as_str() {
                            merged_required.push(s.to_string());
                        }
                    }
                }
            } else {
                merged.insert(k.clone(), v.clone());
            }
        }

        // Then add/override with subschema
        if let Some(obj) = subschema.as_object() {
            for (k, v) in obj {
                // Skip 'not' - we can't represent negation in CFG
                if k == "not" {
                    continue;
                }
                if k == "properties" {
                    if let Some(props) = v.as_object() {
                        for (pk, pv) in props {
                            merged_props.insert(pk.clone(), pv.clone());
                        }
                    }
                } else if k == "required" {
                    if let Some(req) = v.as_array() {
                        for r in req {
                            if let Some(s) = r.as_str() {
                                merged_required.push(s.to_string());
                            }
                        }
                    }
                } else {
                    merged.insert(k.clone(), v.clone());
                }
            }
        }

        if !merged_props.is_empty() {
            merged.insert("properties".to_string(), Value::Object(merged_props));
        }
        if !merged_required.is_empty() {
            let unique: Vec<Value> = merged_required.into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .map(Value::String)
                .collect();
            merged.insert("required".to_string(), Value::Array(unique));
        }

        Ok(Value::Object(merged))
    }

    /// Convert a JSON value to a grammar literal.
    fn value_to_literal(&self, val: &Value) -> GrammarExpr {
        match val {
            Value::Null => GrammarExpr::Literal(b"null".to_vec()),
            Value::Bool(true) => GrammarExpr::Literal(b"true".to_vec()),
            Value::Bool(false) => GrammarExpr::Literal(b"false".to_vec()),
            Value::Number(n) => GrammarExpr::Literal(n.to_string().into_bytes()),
            Value::String(s) => {
                let escaped = self.escape_string_for_json(s);
                GrammarExpr::Literal(format!("\"{}\"", escaped).into_bytes())
            }
            Value::Array(_) | Value::Object(_) => {
                // Serialize to compact JSON
                let json_str = serde_json::to_string(val).unwrap_or_default();
                GrammarExpr::Literal(json_str.into_bytes())
            }
        }
    }

    /// Escape a string for use in JSON.
    fn escape_string_for_json(&self, s: &str) -> String {
        let mut result = String::new();
        for c in s.chars() {
            match c {
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                c if c.is_control() => {
                    result.push_str(&format!("\\u{:04x}", c as u32));
                }
                _ => result.push(c),
            }
        }
        result
    }

    /// Add primitive JSON grammar rules.
    fn add_primitive_rules(&mut self) {
        // Whitespace - can be disabled via SEP1_NO_JSON_WHITESPACE=1
        let no_whitespace = std::env::var("SEP1_NO_JSON_WHITESPACE")
            .map(|v| v == "1")
            .unwrap_or(false);
        
        if no_whitespace {
            // Empty WS - no whitespace allowed
            self.add_rule("WS".to_string(), GrammarExpr::Sequence(vec![]));
        } else {
            self.add_rule("WS".to_string(), GrammarExpr::Repeat(Box::new(
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b" ".to_vec()),
                    GrammarExpr::Literal(b"\t".to_vec()),
                    GrammarExpr::Literal(b"\n".to_vec()),
                    GrammarExpr::Literal(b"\r".to_vec()),
                ])
            )));
        }

        // JSON string
        self.add_rule("JSON_STRING".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            GrammarExpr::Ref("STRING_CHARS".to_string()),
            GrammarExpr::Literal(b"\"".to_vec()),
        ]));

        self.add_rule("STRING_CHARS".to_string(), GrammarExpr::Repeat(Box::new(
            GrammarExpr::Choice(vec![
                GrammarExpr::Ref("STRING_CHAR".to_string()),
                GrammarExpr::Ref("ESCAPE_SEQ".to_string()),
            ])
        )));

        // Any char except ", \, or control chars
        self.add_rule("STRING_CHAR".to_string(), GrammarExpr::CharClass("[^\"\\\\\\x00-\\x1f]".to_string()));

        // Escape sequences
        self.add_rule("ESCAPE_SEQ".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\\".to_vec()),
            GrammarExpr::Choice(vec![
                GrammarExpr::CharClass("[\"\\\\/bfnrt]".to_string()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"u".to_vec()),
                    GrammarExpr::Ref("HEX".to_string()),
                    GrammarExpr::Ref("HEX".to_string()),
                    GrammarExpr::Ref("HEX".to_string()),
                    GrammarExpr::Ref("HEX".to_string()),
                ]),
            ]),
        ]));

        self.add_rule("HEX".to_string(), GrammarExpr::CharClass("[0-9a-fA-F]".to_string()));

        // JSON integer
        self.add_rule("JSON_INTEGER".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"-".to_vec()))),
            GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"0".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::CharClass("[1-9]".to_string()),
                    GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass("[0-9]".to_string()))),
                ]),
            ]),
        ]));

        // JSON number
        self.add_rule("JSON_NUMBER".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Ref("JSON_INTEGER".to_string()),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b".".to_vec()),
                GrammarExpr::Ref("DIGITS".to_string()),
            ]))),
            GrammarExpr::Optional(Box::new(GrammarExpr::Ref("EXPONENT".to_string()))),
        ]));

        self.add_rule("DIGITS".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::CharClass("[0-9]".to_string()),
            GrammarExpr::Repeat(Box::new(GrammarExpr::CharClass("[0-9]".to_string()))),
        ]));

        self.add_rule("EXPONENT".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::CharClass("[eE]".to_string()),
            GrammarExpr::Optional(Box::new(GrammarExpr::CharClass("[+-]".to_string()))),
            GrammarExpr::Ref("DIGITS".to_string()),
        ]));

        // JSON bool
        self.add_rule("JSON_BOOL".to_string(), GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"true".to_vec()),
            GrammarExpr::Literal(b"false".to_vec()),
        ]));

        // JSON null
        self.add_rule("JSON_NULL".to_string(), GrammarExpr::Literal(b"null".to_vec()));

        // Only add the mutually recursive _json_* rules if any of them are needed.
        // These rules form a cycle: _json_value <-> _json_object <-> _json_kv
        //                           _json_value <-> _json_array
        // So if any is needed, all are needed.
        let needs_recursive_json = self.needs_json_value || self.needs_json_object || 
                                   self.needs_json_array || self.needs_json_kv;
        
        if !needs_recursive_json {
            return; // Skip adding recursive rules - they're not used!
        }

        // Generic JSON value
        self.add_rule("_json_value".to_string(), GrammarExpr::Choice(vec![
            GrammarExpr::Ref("_json_object".to_string()),
            GrammarExpr::Ref("_json_array".to_string()),
            GrammarExpr::Ref("JSON_STRING".to_string()),
            GrammarExpr::Ref("JSON_NUMBER".to_string()),
            GrammarExpr::Ref("JSON_BOOL".to_string()),
            GrammarExpr::Ref("JSON_NULL".to_string()),
        ]));

        // Generic JSON object: { (kv (, kv)*)? }
        // WS is handled by the ignore terminal
        let comma_kv = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b",".to_vec()),
            GrammarExpr::Ref("_json_kv".to_string()),
        ]);

        self.add_rule("_json_object".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("_json_kv".to_string()),
                GrammarExpr::Repeat(Box::new(comma_kv)),
            ]))),
            GrammarExpr::Literal(b"}".to_vec()),
        ]));

        // JSON key-value pair
        // WS is handled by the ignore terminal
        self.add_rule("_json_kv".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Ref("JSON_STRING".to_string()),
            GrammarExpr::Literal(b":".to_vec()),
            GrammarExpr::Ref("_json_value".to_string()),
        ]));

        // Generic JSON array
        // WS is handled by the ignore terminal
        let comma_val = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b",".to_vec()),
            GrammarExpr::Ref("_json_value".to_string()),
        ]);

        self.add_rule("_json_array".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"[".to_vec()),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("_json_value".to_string()),
                GrammarExpr::Repeat(Box::new(comma_val)),
            ]))),
            GrammarExpr::Literal(b"]".to_vec()),
        ]));
    }
}

/// Convert a JSON Schema string to EBNF string.
pub fn json_schema_to_ebnf(schema_json: &str) -> Result<String, String> {
    let schema: Value = serde_json::from_str(schema_json)
        .map_err(|e| format!("Failed to parse JSON schema: {}", e))?;
    
    let (rules, root_rule) = JsonSchemaConverter::new(schema).convert()?;
    
    // Check if whitespace is disabled
    let no_whitespace = std::env::var("SEP1_NO_JSON_WHITESPACE")
        .map(|v| v == "1")
        .unwrap_or(false);
    
    // Convert rules to EBNF format
    // Only add ignore directive if whitespace is enabled
    let ignore_prefix = if no_whitespace { "" } else { "#![ignore(WS)]\n\n" };
    let mut ebnf = String::from(ignore_prefix);
    let prefix_len = ignore_prefix.len();
    
    for (name, expr) in &rules {
        if name == &root_rule {
            // Put root rule first (after ignore directive if present)
            ebnf = format!("{}{} ::= {} ;\n", ignore_prefix, name, grammar_expr_to_ebnf(expr)) + &ebnf[prefix_len..];
        } else {
            ebnf.push_str(&format!("{} ::= {} ;\n", name, grammar_expr_to_ebnf(expr)));
        }
    }
    
    Ok(ebnf)
}

/// Convert a JSON Schema to a Vec<(String, GrammarExpr)>.
pub fn json_schema_to_grammar_exprs(schema_json: &str) -> Result<Vec<(String, GrammarExpr)>, String> {
    let schema: Value = serde_json::from_str(schema_json)
        .map_err(|e| format!("Failed to parse JSON schema: {}", e))?;
    
    let (rules, _root_rule) = JsonSchemaConverter::new(schema).convert()?;
    Ok(rules)
}

/// Convert a GrammarExpr to EBNF string.
fn grammar_expr_to_ebnf(expr: &GrammarExpr) -> String {
    match expr {
        GrammarExpr::Ref(name) => name.clone(),
        GrammarExpr::Literal(bytes) => {
            let s = String::from_utf8_lossy(bytes);
            // Escape special characters for EBNF literal
            let mut escaped = String::new();
            for c in s.chars() {
                match c {
                    '\\' => escaped.push_str("\\\\"),
                    '\'' => escaped.push_str("\\'"),
                    '\n' => escaped.push_str("\\n"),
                    '\r' => escaped.push_str("\\r"),
                    '\t' => escaped.push_str("\\t"),
                    c if c.is_control() => {
                        escaped.push_str(&format!("\\x{:02x}", c as u32));
                    }
                    _ => escaped.push(c),
                }
            }
            format!("'{}'", escaped)
        }
        GrammarExpr::Sequence(exprs) => {
            let parts: Vec<String> = exprs.iter().map(grammar_expr_to_ebnf).collect();
            parts.join(" ")
        }
        GrammarExpr::Choice(exprs) => {
            let parts: Vec<String> = exprs.iter().map(|e| grammar_expr_to_ebnf(e)).collect();
            format!("( {} )", parts.join(" | "))
        }
        GrammarExpr::Optional(e) => {
            format!("( {} )?", grammar_expr_to_ebnf(e))
        }
        GrammarExpr::Repeat(e) => {
            format!("( {} )*", grammar_expr_to_ebnf(e))
        }
        GrammarExpr::CharClass(s) => s.clone(),
        GrammarExpr::AnyChar => ".".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_object() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).unwrap();
        assert!(ebnf.contains("root"));
        assert!(ebnf.contains("JSON_STRING"));
        assert!(ebnf.contains("JSON_INTEGER"));
    }

    #[test]
    fn test_any_of() {
        let schema = r#"{
            "anyOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        }"#;
        
        let rules = json_schema_to_grammar_exprs(schema).unwrap();
        // Should have root rule and alternatives
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_enum() {
        let schema = r#"{
            "enum": ["red", "green", "blue"]
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).unwrap();
        assert!(ebnf.contains("\"red\""));
        assert!(ebnf.contains("\"green\""));
        assert!(ebnf.contains("\"blue\""));
    }

    #[test]
    fn test_ref() {
        let schema = r##"{
            "$defs": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            },
            "type": "array",
            "items": {"$ref": "#/$defs/person"}
        }"##;
        
        let rules = json_schema_to_grammar_exprs(schema).unwrap();
        assert!(!rules.is_empty());
    }
}
