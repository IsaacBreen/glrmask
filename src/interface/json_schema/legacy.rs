//! JSON Schema to Grammar conversion (Legacy Implementation).
//!
//! **NOTE:** This is the legacy monolithic implementation. For new development,
//! consider using the modular implementation in:
//! - `json_schema_types`: Intermediate representations (SchemaType, GrammarType)
//! - `json_schema_parser`: JSON -> SchemaType conversion  
//! - `json_schema_convert`: SchemaType -> GrammarType conversion
//! - `json_schema_emit`: GrammarType -> GrammarExpr conversion
//!
//! This legacy module is kept for backward compatibility and because it handles
//! some edge cases that haven't been migrated yet.
//!
//! ---
//!
//! This module converts JSON Schema (draft-07 compatible) to Sep1's grammar representation.
//! The goal is to generate grammars that are as permissive as the schema allows, without
//! trying to enforce semantic constraints that are impossible for CFGs.
//!
//! # Assumptions and Limitations
//!
//! ## Properties are Order-Dependent
//! Properties in objects must appear in the order they are declared in the schema.
//! This is necessary for grammar disambiguation and differs from standard JSON Schema
//! which allows properties in any order.
//!
//! ## Additional Properties Location
//! When `additionalProperties` is enabled, additional properties can ONLY appear
//! AFTER all declared properties. This is a grammar constraint for unambiguous parsing.
//!
//! ## Supported Features
//! - `type`: object, array, string, integer, number, boolean, null, and arrays of types
//! - `properties`, `additionalProperties`
//! - `items`, `prefixItems` (draft 2020-12)
//! - `$ref`, `$defs`, `definitions`
//! - `allOf`, `anyOf`, `oneOf`
//! - `const`, `enum`
//! - `pattern`, `minLength`, `maxLength` (for strings)
//!
//! ## Intentionally Unsupported (semantic constraints, not syntactic)
//! - `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum` (number constraints)
//! - `minItems`/`maxItems` (array length)
//! - `minProperties`/`maxProperties`
//! - `uniqueItems`
//! - `dependencies`, `if`/`then`/`else`, `not`
//! - `format` (stored but not enforced)

use crate::interface::GrammarExpr;
use crate::tokenizer::string_utils::{escape_char_for_char_class, escape_string_for_json};
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
                    "string" => {
                        // Check if there are constraints that need a dedicated rule
                        let has_pattern = obj.get("pattern").is_some();
                        let has_min_length = obj.get("minLength").is_some();
                        let has_max_length = obj.get("maxLength").is_some();
                        if has_pattern || has_min_length || has_max_length {
                            Err("complex".to_string()) // Needs dedicated rule
                        } else {
                            Ok(GrammarExpr::Ref("JSON_STRING".to_string()))
                        }
                    }
                    "integer" => Ok(GrammarExpr::Ref("JSON_INTEGER".to_string())),
                    "number" => Ok(GrammarExpr::Ref("JSON_NUMBER".to_string())),
                    "boolean" => Ok(GrammarExpr::Ref("JSON_BOOL".to_string())),
                    "null" => Ok(GrammarExpr::Ref("JSON_NULL".to_string())),
                    // Object without properties -> generic object
                    "object" => {
                        let has_properties = obj.get("properties").and_then(|v| v.as_object()).map(|p| !p.is_empty()).unwrap_or(false);
                        let has_pattern_props = obj.get("patternProperties").and_then(|v| v.as_object()).map(|p| !p.is_empty()).unwrap_or(false);
                        if !has_properties && !has_pattern_props && obj.get("additionalProperties") != Some(&Value::Bool(false)) {
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
                
                // Check for string constraints that would need a dedicated rule
                let has_pattern = obj.get("pattern").is_some();
                let has_min_length = obj.get("minLength").is_some();
                let has_max_length = obj.get("maxLength").is_some();
                let has_string_constraints = has_pattern || has_min_length || has_max_length;
                
                for t in types {
                    if let Some(type_str) = t.as_str() {
                        match type_str {
                            "string" => {
                                if has_string_constraints {
                                    all_inline = false;
                                    break;
                                }
                                alternatives.push(GrammarExpr::Ref("JSON_STRING".to_string()));
                            }
                            "integer" => alternatives.push(GrammarExpr::Ref("JSON_INTEGER".to_string())),
                            "number" => alternatives.push(GrammarExpr::Ref("JSON_NUMBER".to_string())),
                            "boolean" => alternatives.push(GrammarExpr::Ref("JSON_BOOL".to_string())),
                            "null" => alternatives.push(GrammarExpr::Ref("JSON_NULL".to_string())),
                            "object" => {
                                let has_props = obj.get("properties").and_then(|v| v.as_object()).map(|p| !p.is_empty()).unwrap_or(false);
                                let has_pattern_props = obj.get("patternProperties").and_then(|v| v.as_object()).map(|p| !p.is_empty()).unwrap_or(false);
                                if !has_props && !has_pattern_props && obj.get("additionalProperties") != Some(&Value::Bool(false)) {
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
                // Extract string constraints
                let pattern = obj.get("pattern").and_then(|v| v.as_str()).map(|s| s.to_string());
                let min_length = obj.get("minLength").and_then(|v| v.as_u64());
                let max_length = obj.get("maxLength").and_then(|v| v.as_u64());
                
                // Check if we have any constraints
                if pattern.is_some() || min_length.is_some() || max_length.is_some() {
                    self.convert_string_with_constraints(pattern, min_length, max_length, rule_name)
                } else {
                    self.add_rule(rule_name.clone(), GrammarExpr::Ref("JSON_STRING".to_string()));
                    Ok(rule_name)
                }
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

    /// Convert a string with pattern and/or length constraints.
    fn convert_string_with_constraints(
        &mut self, 
        pattern: Option<String>,
        min_length: Option<u64>,
        max_length: Option<u64>,
        rule_name: String
    ) -> Result<String, String> {
        // For now, if we have length constraints but no pattern, create a length-constrained string
        // If we have a pattern, we need to handle both together
        
        // Create the string content rule
        let content_rule = self.new_rule_name("strcontent");
        
        if let Some(pattern) = pattern {
            // Build pattern-based content
            let pattern_content = self.build_pattern_content(&pattern, min_length, max_length);
            self.add_rule(content_rule.clone(), pattern_content);
        } else {
            // Length-only constraints (no pattern)
            let length_content = self.build_length_constrained_content(min_length, max_length);
            self.add_rule(content_rule.clone(), length_content);
        }
        
        // Final rule: '"' content '"'
        self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            GrammarExpr::Ref(content_rule),
            GrammarExpr::Literal(b"\"".to_vec()),
        ]));
        
        Ok(rule_name)
    }
    
    /// Build content expression for a string with length constraints only.
    fn build_length_constrained_content(&mut self, min_length: Option<u64>, max_length: Option<u64>) -> GrammarExpr {
        // STRING_CHAR_OR_ESCAPE = STRING_CHAR | ESCAPE_SEQ
        let char_or_escape = GrammarExpr::Choice(vec![
            GrammarExpr::Ref("STRING_CHAR".to_string()),
            GrammarExpr::Ref("ESCAPE_SEQ".to_string()),
        ]);
        
        match (min_length, max_length) {
            (None, None) => {
                // No constraints - just STRING_CHARS
                GrammarExpr::Ref("STRING_CHARS".to_string())
            }
            (Some(min), None) => {
                // At least min characters
                // char{min,} = char{min} char*
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..min {
                    parts.push(char_or_escape.clone());
                }
                parts.push(GrammarExpr::Ref("STRING_CHARS".to_string()));
                GrammarExpr::Sequence(parts)
            }
            (None, Some(max)) => {
                // At most max characters
                // char{0,max} = char? char? ... (max times)
                let mut result = GrammarExpr::Sequence(vec![]);
                for _ in 0..max {
                    result = GrammarExpr::Sequence(vec![
                        result,
                        GrammarExpr::Optional(Box::new(char_or_escape.clone())),
                    ]);
                }
                result
            }
            (Some(min), Some(max)) => {
                if max < min {
                    // Invalid constraint - return empty
                    return GrammarExpr::Sequence(vec![]);
                }
                // Exactly between min and max characters
                // char{min,max} = char{min} char{0,max-min}
                let mut parts: Vec<GrammarExpr> = Vec::new();
                // First min required characters
                for _ in 0..min {
                    parts.push(char_or_escape.clone());
                }
                // Then (max-min) optional characters
                for _ in 0..(max - min) {
                    parts.push(GrammarExpr::Optional(Box::new(char_or_escape.clone())));
                }
                GrammarExpr::Sequence(parts)
            }
        }
    }
    
    /// Build content expression for a string with pattern (and optional length constraints).
    fn build_pattern_content(&mut self, pattern: &str, min_length: Option<u64>, max_length: Option<u64>) -> GrammarExpr {
        // Determine if pattern is anchored
        let starts_anchored = pattern.starts_with('^');
        let ends_anchored = pattern.ends_with('$');
        
        // Strip anchors for the inner pattern
        let mut inner = pattern;
        if starts_anchored {
            inner = &inner[1..];
        }
        if ends_anchored && !inner.is_empty() {
            inner = &inner[..inner.len()-1];
        }
        
        // For fully anchored patterns with length constraints, try to apply them
        if starts_anchored && ends_anchored && (min_length.is_some() || max_length.is_some()) {
            // Check if the pattern is a simple character class with + or *
            // e.g., [a-zA-Z]+, \d+, [0-9a-fA-F]+
            if let Some(content) = self.apply_length_to_simple_pattern(inner, min_length, max_length) {
                return content;
            }
        }
        
        // For NON-anchored patterns with length constraints, prioritize length constraints
        // The pattern match is a "soft" constraint that we can't fully enforce for search patterns
        if !starts_anchored || !ends_anchored {
            if min_length.is_some() || max_length.is_some() {
                // Use length constraints only - pattern is too loose to combine
                return self.build_length_constrained_content(min_length, max_length);
            }
        }
        
        // Convert the pattern to an EBNF-compatible format
        let ebnf_pattern = self.regex_to_ebnf_pattern(inner);
        
        // Check if pattern has top-level alternation and needs wrapping
        let pattern_has_alternation = ebnf_pattern.split_whitespace().any(|p| p == "|");
        let pattern_part = if pattern_has_alternation {
            format!("({})", ebnf_pattern)
        } else {
            ebnf_pattern
        };
        
        // Build the full string pattern
        let string_content = if starts_anchored && ends_anchored {
            // Full match: ^pattern$
            pattern_part
        } else if starts_anchored {
            // Anchored at start: ^pattern.*
            format!("{} STRING_CHARS", pattern_part)
        } else if ends_anchored {
            // Anchored at end: .*pattern$
            format!("STRING_CHARS {}", pattern_part)
        } else {
            // Search pattern: .*pattern.*
            format!("STRING_CHARS {} STRING_CHARS", pattern_part)
        };
        
        // Parse and build the pattern expression
        if string_content.trim().is_empty() {
            // Empty pattern - match empty string content
            GrammarExpr::Sequence(vec![])
        } else {
            // Create the pattern as a sequence/choice structure
            self.build_pattern_expr(&string_content)
        }
    }
    
    /// Try to apply length constraints to a simple regex pattern.
    /// Returns Some(expr) if successful, None if the pattern is too complex.
    fn apply_length_to_simple_pattern(&mut self, pattern: &str, min_length: Option<u64>, max_length: Option<u64>) -> Option<GrammarExpr> {
        // Handle patterns like:
        // - [a-zA-Z]+  -> [a-zA-Z]{min,max}
        // - [0-9]*     -> [0-9]{min,max}  
        // - \d+        -> [0-9]{min,max}
        // - \w+        -> [a-zA-Z0-9_]{min,max}
        
        let trimmed = pattern.trim();
        
        // Check for patterns ending in + or *
        if !trimmed.ends_with('+') && !trimmed.ends_with('*') {
            return None;
        }
        
        let base_pattern = &trimmed[..trimmed.len()-1];
        let is_plus = trimmed.ends_with('+');
        
        // Parse the base pattern to get a character class
        let char_class = if base_pattern.starts_with('[') && base_pattern.ends_with(']') {
            // It's already a character class - sanitize for EBNF
            self.sanitize_char_class(base_pattern)
        } else if base_pattern == "\\d" {
            "[0-9]".to_string()
        } else if base_pattern == "\\D" {
            "[^0-9]".to_string()
        } else if base_pattern == "\\w" {
            "[a-zA-Z0-9_]".to_string()
        } else if base_pattern == "\\W" {
            "[^a-zA-Z0-9_]".to_string()
        } else if base_pattern == "\\s" {
            "[ \\t\\n\\r]".to_string()
        } else if base_pattern == "\\S" {
            "[^ \\t\\n\\r]".to_string()
        } else if base_pattern == "." {
            // Any char - use STRING_CHAR | ESCAPE_SEQ
            return self.apply_length_to_any_char(min_length, max_length, is_plus);
        } else {
            return None;
        };
        
        let char_expr = GrammarExpr::CharClass(char_class);
        
        // Calculate effective min/max
        let min = if is_plus {
            std::cmp::max(min_length.unwrap_or(1), 1)
        } else {
            min_length.unwrap_or(0)
        };
        let max = max_length;
        
        match (min, max) {
            (0, None) => {
                // Zero or more
                Some(GrammarExpr::Repeat(Box::new(char_expr)))
            }
            (min, None) if min > 0 => {
                // At least min
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..min {
                    parts.push(char_expr.clone());
                }
                parts.push(GrammarExpr::Repeat(Box::new(char_expr)));
                Some(GrammarExpr::Sequence(parts))
            }
            (0, Some(max)) => {
                // At most max
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..max {
                    parts.push(GrammarExpr::Optional(Box::new(char_expr.clone())));
                }
                Some(GrammarExpr::Sequence(parts))
            }
            (min, Some(max)) if max >= min => {
                // Between min and max
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..min {
                    parts.push(char_expr.clone());
                }
                for _ in 0..(max - min) {
                    parts.push(GrammarExpr::Optional(Box::new(char_expr.clone())));
                }
                Some(GrammarExpr::Sequence(parts))
            }
            _ => None
        }
    }
    
    /// Sanitize a character class for EBNF compatibility.
    /// EBNF character classes can't contain literal (, ), {, }, [ or ] without escaping.
    fn sanitize_char_class(&self, class_def: &str) -> String {
        // Strip [ and ]
        let content = &class_def[1..class_def.len()-1];
        
        // Check for negation
        let (negated, content) = if content.starts_with('^') {
            (true, &content[1..])
        } else {
            (false, content)
        };
        
        let mut result = String::from("[");
        if negated {
            result.push('^');
        }
        
        let mut chars = content.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    // Escape sequence - pass through
                    result.push(c);
                    if let Some(next) = chars.next() {
                        result.push(next);
                    }
                }
                '(' | ')' | '{' | '}' | '[' | ']' => {
                    // These need escaping for EBNF
                    result.push('\\');
                    result.push(c);
                }
                _ => result.push(c),
            }
        }
        
        result.push(']');
        result
    }
    
    /// Apply length constraints to "any character" pattern (.).
    fn apply_length_to_any_char(&mut self, min_length: Option<u64>, max_length: Option<u64>, is_plus: bool) -> Option<GrammarExpr> {
        let char_or_escape = GrammarExpr::Choice(vec![
            GrammarExpr::Ref("STRING_CHAR".to_string()),
            GrammarExpr::Ref("ESCAPE_SEQ".to_string()),
        ]);
        
        let min = if is_plus {
            std::cmp::max(min_length.unwrap_or(1), 1)
        } else {
            min_length.unwrap_or(0)
        };
        let max = max_length;
        
        match (min, max) {
            (0, None) => {
                Some(GrammarExpr::Ref("STRING_CHARS".to_string()))
            }
            (min, None) if min > 0 => {
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..min {
                    parts.push(char_or_escape.clone());
                }
                parts.push(GrammarExpr::Ref("STRING_CHARS".to_string()));
                Some(GrammarExpr::Sequence(parts))
            }
            (0, Some(max)) => {
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..max {
                    parts.push(GrammarExpr::Optional(Box::new(char_or_escape.clone())));
                }
                Some(GrammarExpr::Sequence(parts))
            }
            (min, Some(max)) if max >= min => {
                let mut parts: Vec<GrammarExpr> = Vec::new();
                for _ in 0..min {
                    parts.push(char_or_escape.clone());
                }
                for _ in 0..(max - min) {
                    parts.push(GrammarExpr::Optional(Box::new(char_or_escape.clone())));
                }
                Some(GrammarExpr::Sequence(parts))
            }
            _ => None
        }
    }

    /// Convert a regex pattern fragment to EBNF-compatible format.
    /// This handles common regex constructs but is not a full regex implementation.
    fn regex_to_ebnf_pattern(&self, pattern: &str) -> String {
        // This is a simplified conversion that handles common patterns
        // Full regex support would require a proper regex-to-CFG conversion
        
        let mut result = String::new();
        let mut chars = pattern.chars().peekable();
        
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    // Escape sequence
                    if let Some(&next) = chars.peek() {
                        match next {
                            'd' => {
                                chars.next();
                                result.push_str("[0-9]");
                            }
                            'D' => {
                                chars.next();
                                result.push_str("[^0-9]");
                            }
                            'w' => {
                                chars.next();
                                result.push_str("[a-zA-Z0-9_]");
                            }
                            'W' => {
                                chars.next();
                                result.push_str("[^a-zA-Z0-9_]");
                            }
                            's' => {
                                chars.next();
                                result.push_str("[ \\t\\n\\r]");
                            }
                            'S' => {
                                chars.next();
                                result.push_str("[^ \\t\\n\\r]");
                            }
                            'n' => {
                                chars.next();
                                // In JSON, newline is encoded as \n (two chars)
                                result.push_str("'\\n'");
                            }
                            'r' => {
                                chars.next();
                                result.push_str("'\\r'");
                            }
                            't' => {
                                chars.next();
                                result.push_str("'\\t'");
                            }
                            '\\' | '/' | '"' | '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$' => {
                                chars.next();
                                // Literal escaped character
                                result.push('\'');
                                result.push(next);
                                result.push('\'');
                            }
                            _ => {
                                // Unknown escape, treat as literal
                                chars.next();
                                result.push('\'');
                                result.push(next);
                                result.push('\'');
                            }
                        }
                    }
                }
                '[' => {
                    // Character class - collect and sanitize for EBNF
                    let mut class = String::from("[");
                    let mut depth = 1;
                    while depth > 0 {
                        if let Some(cc) = chars.next() {
                            // Sanitize characters that need escaping in EBNF char classes
                            if cc == '\\' {
                                class.push(cc);
                                if let Some(esc) = chars.next() {
                                    class.push(esc);
                                }
                            } else if cc == '[' {
                                depth += 1;
                                class.push('\\');
                                class.push(cc);
                            } else if cc == ']' {
                                depth -= 1;
                                if depth > 0 {
                                    // Not the final ], escape it
                                    class.push('\\');
                                    class.push(cc);
                                } else {
                                    class.push(cc);
                                }
                            } else if cc == '(' || cc == ')' || cc == '{' || cc == '}' {
                                // These need escaping for EBNF
                                class.push('\\');
                                class.push(cc);
                            } else {
                                class.push(cc);
                            }
                        } else {
                            break;
                        }
                    }
                    result.push_str(&class);
                }
                '.' => {
                    // Any character except newline
                    // In JSON strings, most printable chars are allowed directly
                    result.push_str("( STRING_CHAR | ESCAPE_SEQ )");
                }
                '*' | '+' => {
                    // Quantifier - append to result
                    result.push(c);
                }
                '?' => {
                    // Could be quantifier or part of (?:) non-capturing group
                    // Just pass through - it will be handled with context
                    result.push(c);
                }
                '(' => {
                    // Grouping - check for non-capturing group (?:...)
                    result.push(c);
                    // Check if this is (?:...) non-capturing group
                    if chars.peek() == Some(&'?') {
                        chars.next(); // consume '?'
                        if chars.peek() == Some(&':') {
                            chars.next(); // consume ':'
                            // Skip the ?:, just output (
                        } else {
                            // Some other ? construct, just skip the ?
                        }
                    }
                }
                ')' => {
                    // Closing group
                    result.push(c);
                }
                '|' => {
                    // Alternation
                    result.push_str(" | ");
                }
                '^' | '$' => {
                    // Anchors - skip them within the pattern (they're handled at a higher level)
                    // When they appear inside groups like (^gs://.+), just skip them
                }
                '{' => {
                    // Quantifier {n} or {n,m} - pass through until }
                    result.push(c);
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        result.push(nc);
                        if nc == '}' { break; }
                    }
                }
                _ => {
                    // Literal character - collect consecutive alphanumerics into single quoted string
                    let mut literal = String::new();
                    literal.push(c);
                    
                    // Peek ahead and collect more literal characters
                    while let Some(&next) = chars.peek() {
                        // Stop at special regex chars
                        if next == '\\' || next == '[' || next == '.' || next == '*' 
                            || next == '+' || next == '?' || next == '(' || next == ')'
                            || next == '|' || next == '{' || next == '^' || next == '$' {
                            break;
                        }
                        literal.push(chars.next().unwrap());
                    }
                    
                    // Output as a single quoted literal, escaping single quotes if any
                    result.push('\'');
                    for lc in literal.chars() {
                        if lc == '\'' {
                            result.push_str("\\'");
                        } else {
                            result.push(lc);
                        }
                    }
                    result.push('\'');
                }
            }
        }
        
        result
    }
    
    /// Build a GrammarExpr from an EBNF-style pattern string.
    /// This parses a simplified EBNF pattern format.
    fn build_pattern_expr(&mut self, pattern: &str) -> GrammarExpr {
        // For now, use CharClass for the whole pattern if it looks like a simple regex
        // More complex patterns would need proper parsing
        
        // Simple approach: If the pattern is a single regex-like expression, use CharClass
        // Otherwise, create a sequence with refs
        
        let trimmed = pattern.trim();
        
        // Check if it's referencing existing rules
        if trimmed == "STRING_CHARS" {
            return GrammarExpr::Ref("STRING_CHARS".to_string());
        }
        
        // Split by whitespace but handle alternation specially
        // The pattern string uses " | " for alternation (with spaces)
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        
        // Check if there's alternation at the top level
        let has_top_level_alternation = parts.iter().any(|p| *p == "|");
        
        if has_top_level_alternation {
            // Split by " | " and build a Choice
            let alternatives: Vec<&str> = trimmed.split(" | ").collect();
            if alternatives.len() > 1 {
                let choice_exprs: Vec<GrammarExpr> = alternatives
                    .iter()
                    .map(|alt| self.build_pattern_expr(alt.trim()))
                    .collect();
                return GrammarExpr::Choice(choice_exprs);
            }
        }
        
        if parts.len() == 1 {
            // Single element
            let part = parts[0];
            if part == "STRING_CHARS" {
                return GrammarExpr::Ref("STRING_CHARS".to_string());
            } else if part.starts_with('[') && part.ends_with(']') {
                // Character class
                return GrammarExpr::CharClass(part.to_string());
            } else if part.starts_with('\'') && part.ends_with('\'') && part.len() >= 2 {
                // Literal
                let inner = &part[1..part.len()-1];
                return GrammarExpr::Literal(inner.as_bytes().to_vec());
            } else if part.starts_with('(') && part.ends_with(')') {
                // Grouped expression - recurse
                return self.build_pattern_expr(&part[1..part.len()-1]);
            } else {
                // Assume it's a reference
                return GrammarExpr::Ref(part.to_string());
            }
        }
        
        // Multiple parts - create a sequence
        let mut seq = Vec::new();
        for part in parts {
            // Skip standalone "|" tokens as they've been handled above
            if part == "|" {
                continue;
            }
            seq.push(self.build_pattern_expr(part));
        }
        
        if seq.len() == 1 {
            seq.remove(0)
        } else {
            GrammarExpr::Sequence(seq)
        }
    }

    /// Generate a GrammarExpr for a JSON string key that is NOT any of the excluded property names.
    /// 
    /// This is used for additionalProperties to ensure they don't match declared property names.
    /// The strategy:
    /// 1. If no exclusions, just return JSON_STRING
    /// 2. Otherwise, generate a pattern that matches strings differing from ALL excluded strings
    ///
    /// For implementation, we build a choice of patterns:
    /// - Strings with different lengths than all excluded strings
    /// - Strings starting with characters not at start of any excluded string
    /// - Strings starting the same as some excluded string but diverging later
    fn json_string_except(&mut self, excluded: &[String]) -> GrammarExpr {
        if excluded.is_empty() {
            return GrammarExpr::Ref("JSON_STRING".to_string());
        }

        // Create a unique rule name for this exclusion pattern
        let rule_name = self.new_rule_name("str_ex");
        
        // Build the exclusion pattern
        let pattern = self.build_string_except_pattern(excluded);
        self.add_rule(rule_name.clone(), pattern);
        
        GrammarExpr::Ref(rule_name)
    }

    /// Build a grammar pattern for strings not matching any of the excluded strings.
    /// The pattern is: `"` + (content NOT matching any excluded) + `"`
    fn build_string_except_pattern(&mut self, excluded: &[String]) -> GrammarExpr {
        // Strategy: Build a trie-like pattern of exclusions
        // A string matches if it differs from ALL excluded strings at some position
        
        // Group excluded strings by first character
        let mut by_first_char: std::collections::BTreeMap<char, Vec<&str>> = std::collections::BTreeMap::new();
        let mut first_chars: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
        
        for s in excluded {
            if let Some(c) = s.chars().next() {
                first_chars.insert(c);
                by_first_char.entry(c).or_default().push(s.as_str());
            }
        }

        // Build alternatives for the content between quotes
        let mut content_alternatives = Vec::new();

        // Alternative 1: Start with a character NOT in first_chars
        // This automatically avoids all excluded strings
        let first_chars_escaped: Vec<String> = first_chars.iter()
            .map(|c| escape_char_for_char_class(*c))
            .collect();
        
        if !first_chars_escaped.is_empty() {
            let not_first_chars_pattern = format!("[^{}\"\\\\\\x00-\\x1f]", first_chars_escaped.join(""));
            // Matches: (not_first_char) + any_remaining_chars
            content_alternatives.push(GrammarExpr::Sequence(vec![
                GrammarExpr::CharClass(not_first_chars_pattern),
                GrammarExpr::Ref("STRING_CHARS".to_string()),
            ]));
        }

        // Alternative 2: For each first char, match strings that START with that char
        // but then DIVERGE from all excluded strings starting with that char
        for (first_char, strings_with_this_first) in &by_first_char {
            let suffixes: Vec<&str> = strings_with_this_first.iter()
                .map(|s| &s[first_char.len_utf8()..])
                .collect();
            
            let divergent_suffix = self.build_divergent_suffix_pattern(&suffixes);
            
            // Pattern: first_char + divergent_suffix
            let first_char_literal = escape_string_for_json(&first_char.to_string());
            content_alternatives.push(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(first_char_literal.into_bytes()),
                divergent_suffix,
            ]));
        }

        // Also allow empty string (if no excluded string is empty)
        if !excluded.iter().any(|s| s.is_empty()) {
            content_alternatives.push(GrammarExpr::Sequence(vec![])); // epsilon
        }

        // Combine all alternatives
        let content = if content_alternatives.len() == 1 {
            content_alternatives.remove(0)
        } else {
            GrammarExpr::Choice(content_alternatives)
        };

        // Final pattern: `"` + content + `"`
        GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"\"".to_vec()),
            content,
            GrammarExpr::Literal(b"\"".to_vec()),
        ])
    }

    /// Build a pattern for suffixes that diverge from all the given suffixes.
    /// Returns a pattern that matches strings NOT equal to any of the suffixes.
    fn build_divergent_suffix_pattern(&self, suffixes: &[&str]) -> GrammarExpr {
        if suffixes.is_empty() {
            // No suffixes to avoid - match anything
            return GrammarExpr::Ref("STRING_CHARS".to_string());
        }

        // If all suffixes are empty, we need to match at least one more character
        if suffixes.iter().all(|s| s.is_empty()) {
            // Must have at least one more character to diverge
            return GrammarExpr::Sequence(vec![
                GrammarExpr::Choice(vec![
                    GrammarExpr::Ref("STRING_CHAR".to_string()),
                    GrammarExpr::Ref("ESCAPE_SEQ".to_string()),
                ]),
                GrammarExpr::Ref("STRING_CHARS".to_string()),
            ]);
        }

        // Group by next character
        let mut by_next_char: std::collections::BTreeMap<char, Vec<&str>> = std::collections::BTreeMap::new();
        let mut next_chars: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
        let mut has_empty = false;
        
        for suffix in suffixes {
            if suffix.is_empty() {
                has_empty = true;
            } else if let Some(c) = suffix.chars().next() {
                next_chars.insert(c);
                by_next_char.entry(c).or_default().push(&suffix[c.len_utf8()..]);
            }
        }

        let mut alternatives = Vec::new();

        // Alternative 1: Use a character not in next_chars
        if !next_chars.is_empty() {
            let chars_escaped: Vec<String> = next_chars.iter()
                .map(|c| escape_char_for_char_class(*c))
                .collect();
            let not_next_chars = format!("[^{}\"\\\\\\x00-\\x1f]", chars_escaped.join(""));
            alternatives.push(GrammarExpr::Sequence(vec![
                GrammarExpr::CharClass(not_next_chars),
                GrammarExpr::Ref("STRING_CHARS".to_string()),
            ]));
        }

        // Alternative 2: For each next char, recursively build divergent pattern
        for (next_char, sub_suffixes) in &by_next_char {
            let next_divergent = self.build_divergent_suffix_pattern(sub_suffixes);
            let next_char_literal = escape_string_for_json(&next_char.to_string());
            alternatives.push(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(next_char_literal.into_bytes()),
                next_divergent,
            ]));
        }

        // Alternative 3: If no empty suffix, we can also stop here (empty continuation)
        if !has_empty {
            alternatives.push(GrammarExpr::Sequence(vec![])); // epsilon - empty string
        }

        if alternatives.len() == 1 {
            alternatives.remove(0)
        } else if alternatives.is_empty() {
            // All suffixes were empty, must have more chars
            GrammarExpr::Sequence(vec![
                GrammarExpr::Choice(vec![
                    GrammarExpr::Ref("STRING_CHAR".to_string()),
                    GrammarExpr::Ref("ESCAPE_SEQ".to_string()),
                ]),
                GrammarExpr::Ref("STRING_CHARS".to_string()),
            ])
        } else {
            GrammarExpr::Choice(alternatives)
        }
    }

    /// Convert an object schema.
    fn convert_object(&mut self, obj: &serde_json::Map<String, Value>, rule_name: String) -> Result<String, String> {
        let properties = obj.get("properties").and_then(|v| v.as_object());
        let additional_props = obj.get("additionalProperties");
        let pattern_props = obj.get("patternProperties").and_then(|v| v.as_object());

        // If no properties defined and additional allowed, just use generic object
        let has_properties = properties.map(|p| !p.is_empty()).unwrap_or(false);
        let has_pattern_properties = pattern_props.map(|p| !p.is_empty()).unwrap_or(false);
        if !has_properties && !has_pattern_properties && additional_props != Some(&Value::Bool(false)) {
            let ref_expr = self.json_object_ref();
            self.add_rule(rule_name.clone(), ref_expr);
            return Ok(rule_name);
        }

        // If no properties, no pattern properties, and no additional allowed, empty object only
        if !has_properties && !has_pattern_properties && additional_props == Some(&Value::Bool(false)) {
            self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"{".to_vec()),
                GrammarExpr::Literal(b"}".to_vec()),
            ]));
            return Ok(rule_name);
        }

        // NEW APPROACH: Order-dependent property matching
        // 
        // Properties must appear in the order they're declared in the schema.
        // Additional properties (if allowed) can only appear AFTER all declared properties.
        // This avoids the ambiguity where `"propName"` could match either the constrained
        // property rule or the generic _json_kv fallback.
        //
        // Example for schema {"properties": {"x": ..., "y": ...}, "additionalProperties": true}:
        //   object ::= '{' 
        //              ( '"x"' ':' xValue ( ',' '"y"' ':' yValue ( ',' _json_kv )* )? )?
        //              | ( '"y"' ':' yValue ( ',' _json_kv )* )?
        //              | ( _json_kv ( ',' _json_kv )* )?
        //              '}'
        //
        // Wait, that's still ambiguous. The key insight is:
        // - Each declared property can optionally appear, in order
        // - Additional properties come after all declared properties
        // - We generate: '{' prop1? (',' prop2)? (',' prop3)? (',' _json_kv)* '}'
        // 
        // Actually simpler: since JSON allows properties in any order, but we need to
        // disambiguate at parse time, we'll generate a grammar where declared properties
        // MUST appear in schema order. Additional properties come after.
        //
        // Grammar: '{' ( declared_members )? ( ',' _json_kv )* '}'
        // declared_members: each property in sequence, all optional with commas between

        let mut sequence_parts = Vec::new();
        sequence_parts.push(GrammarExpr::Literal(b"{".to_vec()));

        // Build the declared properties sequence
        // Properties are processed in the order they appear in the schema.
        // Note: serde_json with preserve_order feature maintains insertion order.
        let mut declared_props = Vec::new();
        if let Some(props) = properties {
            for (prop_name, prop_schema) in props {
                // Convert the property value schema
                let prop_value_expr = match self.convert_schema_inline(prop_schema) {
                    Ok(expr) => expr,
                    Err(_) => {
                        let prop_value_rule = self.new_rule_name("pv");
                        self.convert_schema(prop_schema, prop_value_rule.clone())?;
                        GrammarExpr::Ref(prop_value_rule)
                    }
                };

                let escaped_name = escape_string_for_json(prop_name);
                declared_props.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(format!("\"{}\"", escaped_name).into_bytes()),
                    GrammarExpr::Literal(b":".to_vec()),
                    prop_value_expr,
                ]));
            }
        }

        // Check if we have pattern properties
        let has_pattern_props = pattern_props.map(|p| !p.is_empty()).unwrap_or(false);

        // Determine if additional properties are allowed
        // Note: patternProperties count as "additional" even when additionalProperties: false
        let allow_additional = match additional_props {
            None | Some(Value::Bool(true)) => true,
            Some(Value::Object(_)) => true, // Schema for additional - still allowed
            _ => has_pattern_props, // additionalProperties: false, but patternProperties may exist
        };

        // Collect declared property names for exclusion from additional properties
        let declared_prop_names: Vec<String> = properties
            .map(|p| p.keys().cloned().collect())
            .unwrap_or_default();

        // Build the additional properties expression
        // IMPORTANT: additionalProperties keys must NOT match declared property names
        // This ensures we don't have ambiguity between declared and additional properties
        let additional_kv_expr = match additional_props {
            Some(Value::Object(ap_schema)) => {
                // Constrained additional properties
                let ap_value = Value::Object(ap_schema.clone());
                let ap_expr = match self.convert_schema_inline(&ap_value) {
                    Ok(expr) => expr,
                    Err(_) => {
                        let additional_rule = self.new_rule_name("ap");
                        self.convert_schema(&ap_value, additional_rule.clone())?;
                        GrammarExpr::Ref(additional_rule)
                    }
                };
                // Use string pattern that excludes declared property names
                let key_expr = self.json_string_except(&declared_prop_names);
                Some(GrammarExpr::Sequence(vec![
                    key_expr,
                    GrammarExpr::Literal(b":".to_vec()),
                    ap_expr,
                ]))
            }
            None | Some(Value::Bool(true)) => {
                // Generic additional properties - use exclusion pattern
                let key_expr = self.json_string_except(&declared_prop_names);
                Some(GrammarExpr::Sequence(vec![
                    key_expr,
                    GrammarExpr::Literal(b":".to_vec()),
                    self.json_value_ref(),
                ]))
            }
            _ => {
                // additionalProperties: false - no generic additional allowed
                // (patternProperties handled separately below)
                None
            }
        };

        // Handle patternProperties - add to additional as alternatives
        // Since we can't enforce patterns in CFG, they're treated like additional properties
        // Also exclude declared property names from pattern property keys
        let mut additional_alternatives = Vec::new();
        if let Some(ap_expr) = additional_kv_expr {
            additional_alternatives.push(ap_expr);
        }
        if let Some(pp) = pattern_props {
            for (_pattern, pp_schema) in pp {
                let pp_expr = match self.convert_schema_inline(pp_schema) {
                    Ok(expr) => expr,
                    Err(_) => {
                        let pp_rule = self.new_rule_name("pp");
                        self.convert_schema(pp_schema, pp_rule.clone())?;
                        GrammarExpr::Ref(pp_rule)
                    }
                };
                // Use exclusion pattern for pattern property keys too
                let key_expr = self.json_string_except(&declared_prop_names);
                additional_alternatives.push(GrammarExpr::Sequence(vec![
                    key_expr,
                    GrammarExpr::Literal(b":".to_vec()),
                    pp_expr,
                ]));
            }
        }

        let additional_member = if additional_alternatives.is_empty() {
            None
        } else if additional_alternatives.len() == 1 {
            Some(additional_alternatives.remove(0))
        } else {
            Some(GrammarExpr::Choice(additional_alternatives))
        };

        // Build the object body
        // Strategy: Generate all permutations of declared properties (since JSON allows any order)
        // but that's exponential. Instead, we'll use a simpler approach:
        //
        // Generate: '{' ( any_declared_or_additional ( ',' any_declared_or_additional )* )? '}'
        // where any_declared_or_additional = declared_prop_1 | declared_prop_2 | ... | additional
        //
        // BUT this has the original ambiguity problem!
        //
        // NEW STRATEGY: Require declared properties in order, with optional skipping
        // '{' prop1? (',' prop2?)* (',' additional)* '}'
        //
        // Even simpler: treat each property as optional, in sequence, then allow additional
        // But we need to handle comma separators correctly.
        //
        // SIMPLEST WORKING APPROACH for now:
        // If there are N declared properties, generate:
        //   '{' ( prop0 (',' prop1 (',' prop2 ... (',' additional)*...)?)?)?
        //     | ( prop1 (',' prop2 ... (',' additional)*...)?)?
        //     | ( prop2 ... (',' additional)*...)?
        //     | ( additional (',' additional)* )?
        //   '}'
        //
        // This allows any subset of properties in order, with additional at the end.
        // Each alternative is unambiguous because the first token disambiguates.

        if declared_props.is_empty() {
            // No declared properties, only additional
            if let Some(ref am) = additional_member {
                let comma_additional = GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    am.clone(),
                ]);
                let members = GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    am.clone(),
                    GrammarExpr::Repeat(Box::new(comma_additional)),
                ])));
                self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"{".to_vec()),
                    members,
                    GrammarExpr::Literal(b"}".to_vec()),
                ]));
            } else {
                // Empty object only
                self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"{".to_vec()),
                    GrammarExpr::Literal(b"}".to_vec()),
                ]));
            }
            return Ok(rule_name);
        }

        // Generate ordered property sequence using a LINEAR approach
        // 
        // KEY INSIGHT from llguidance: Use NAMED RULES for suffixes to avoid duplication.
        // Instead of inlining `after_rules[j]` which causes exponential growth,
        // we create named rules and reference them.
        //
        // Grammar structure:
        //   object := '{' _obj_start_or_empty '}'
        //   _obj_start_or_empty := (prop_0 _after_0 | prop_1 _after_1 | ... | prop_n _after_n | additional_start)?
        //   _after_i := (',' (prop_j _after_j | ... | additional_start))?  for each j > i
        //   additional_start := additional (',' additional)*
        //
        // This creates O(n) rules, not O(2^n) expressions.
        
        // Generate unique rule name prefix for this object
        let obj_prefix = self.new_rule_name("obj");
        
        // Create property rules: _obj_prefix_prop_i for each property
        let mut prop_rule_names = Vec::new();
        for (i, prop_expr) in declared_props.iter().enumerate() {
            let prop_rule_name = format!("{}_p{}", obj_prefix, i);
            self.add_rule(prop_rule_name.clone(), prop_expr.clone());
            prop_rule_names.push(prop_rule_name);
        }
        
        // Create the additional_start rule: additional (',' additional)*
        let additional_start_rule = if let Some(ref am) = additional_member {
            let rule_name = format!("{}_ap", obj_prefix);
            let comma_additional = GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b",".to_vec()),
                am.clone(),
            ]);
            self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
                am.clone(),
                GrammarExpr::Repeat(Box::new(comma_additional)),
            ]));
            Some(rule_name)
        } else {
            None
        };
        
        // Create _after_i rules (backwards, so we can reference later ones)
        // _after_i represents what can come after property i
        let mut after_rule_names = Vec::new();
        for i in 0..declared_props.len() {
            after_rule_names.push(format!("{}_a{}", obj_prefix, i));
        }
        
        // Define _after_n-1 (after the last property): just additional tail
        let last_idx = declared_props.len() - 1;
        let last_after_expr = if let Some(ref ap_rule) = additional_start_rule {
            // Can continue with ',' + additional_start, or nothing
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b",".to_vec()),
                GrammarExpr::Ref(ap_rule.clone()),
            ])))
        } else {
            GrammarExpr::Sequence(vec![]) // epsilon - nothing can follow
        };
        self.add_rule(after_rule_names[last_idx].clone(), last_after_expr);
        
        // Define _after_i for i < n-1 (backwards)
        for i in (0..last_idx).rev() {
            // Options: ',' followed by (prop_j _after_j for j > i) OR additional_start
            let mut options = Vec::new();
            
            // Option: go to each later property
            for j in (i + 1)..declared_props.len() {
                options.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(prop_rule_names[j].clone()),
                    GrammarExpr::Ref(after_rule_names[j].clone()),
                ]));
            }
            
            // Option: switch to additional properties
            if let Some(ref ap_rule) = additional_start_rule {
                options.push(GrammarExpr::Ref(ap_rule.clone()));
            }
            
            let continuation = if options.len() == 1 {
                options.remove(0)
            } else if options.is_empty() {
                // No continuation possible - _after_i is epsilon
                self.add_rule(after_rule_names[i].clone(), GrammarExpr::Sequence(vec![]));
                continue;
            } else {
                GrammarExpr::Choice(options)
            };
            
            // _after_i := (',' continuation)?
            self.add_rule(after_rule_names[i].clone(), GrammarExpr::Optional(Box::new(
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b",".to_vec()),
                    continuation,
                ])
            )));
        }
        
        // Build object start: can start with any declared property, or additional only
        let mut start_options = Vec::new();
        for i in 0..declared_props.len() {
            start_options.push(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref(prop_rule_names[i].clone()),
                GrammarExpr::Ref(after_rule_names[i].clone()),
            ]));
        }
        
        // Add additional-properties-only start option
        // 
        // This is now safe because we use json_string_except() to generate exclusion
        // patterns for additional property keys. The exclusion pattern ensures that
        // additional property keys CANNOT match declared property names.
        //
        // For example, if declared properties are "foo" and "bar", the additional
        // property key pattern matches any JSON string EXCEPT "foo" and "bar".
        // This means there's no ambiguity - the first token unambiguously determines
        // whether we're looking at a declared property or an additional property.
        if let Some(ref ap_rule) = additional_start_rule {
            start_options.push(GrammarExpr::Ref(ap_rule.clone()));
        }

        let members = if start_options.len() == 1 {
            GrammarExpr::Optional(Box::new(start_options.remove(0)))
        } else {
            GrammarExpr::Optional(Box::new(GrammarExpr::Choice(start_options)))
        };

        self.add_rule(rule_name.clone(), GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            members,
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
    /// Convert a JSON value to a grammar literal.
    fn value_to_literal(&self, val: &Value) -> GrammarExpr {
        match val {
            Value::Null => GrammarExpr::Literal(b"null".to_vec()),
            Value::Bool(true) => GrammarExpr::Literal(b"true".to_vec()),
            Value::Bool(false) => GrammarExpr::Literal(b"false".to_vec()),
            Value::Number(n) => GrammarExpr::Literal(n.to_string().into_bytes()),
            Value::String(s) => {
                let escaped = escape_string_for_json(s);
                GrammarExpr::Literal(format!("\"{}\"", escaped).into_bytes())
            }
            Value::Array(items) => {
                let mut parts = Vec::new();
                parts.push(GrammarExpr::Literal(b"[".to_vec()));
                
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        parts.push(GrammarExpr::Literal(b",".to_vec()));
                    }
                    parts.push(self.value_to_literal(item));
                }
                
                parts.push(GrammarExpr::Literal(b"]".to_vec()));
                GrammarExpr::Sequence(parts)
            }
            Value::Object(map) => {
                let mut parts = Vec::new();
                parts.push(GrammarExpr::Literal(b"{".to_vec()));
                
                for (i, (key, value)) in map.iter().enumerate() {
                    if i > 0 {
                        parts.push(GrammarExpr::Literal(b",".to_vec()));
                    }
                    
                    // Key string
                    let escaped_key = escape_string_for_json(key);
                    parts.push(GrammarExpr::Literal(format!("\"{}\"", escaped_key).into_bytes()));
                    
                    // Colon
                    parts.push(GrammarExpr::Literal(b":".to_vec()));
                    
                    // Value
                    parts.push(self.value_to_literal(value));
                }
                
                parts.push(GrammarExpr::Literal(b"}".to_vec()));
                GrammarExpr::Sequence(parts)
            }
        }
    }

    /// Add primitive JSON grammar rules.
    fn add_primitive_rules(&mut self) {
        // Whitespace - can be disabled via SEP1_NO_JSON_WHITESPACE=1
        let no_whitespace = std::env::var("SEP1_NO_JSON_WHITESPACE")
            .map(|v| v == "1")
            .unwrap_or(false);
        
        if !no_whitespace {
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
