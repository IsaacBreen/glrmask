//! Schema to Grammar Conversion
//!
//! Converts SchemaType to GrammarType - the second stage of the pipeline.
//!
//! # Key Design Decisions
//!
//! ## Property Ordering
//! Properties are emitted in declaration order. This is crucial for disambiguation
//! in the grammar. When additionalProperties is allowed, they can only appear
//! AFTER all declared properties.
//!
//! ## additionalProperties Handling
//! - `additionalProperties: false` - Only declared properties allowed
//! - `additionalProperties: true` - After declared props, any key-value pairs allowed
//! - `additionalProperties: {schema}` - After declared props, constrained k-v pairs allowed
//!
//! ## Array Items
//! - `items: {schema}` - All items match the schema (homogeneous)
//! - `prefixItems: [...]` - First N items must match specific schemas
//! - Combination: prefix items first, then items schema for rest

use super::types::*;
use crate::dfa_u8::string_utils::escape_string_for_json;
use serde_json::Value;
use std::collections::BTreeMap;

/// Context for schema-to-grammar conversion
pub struct SchemaToGrammar {
    /// Counter for generating unique rule names
    rule_counter: usize,
    /// Generated rules (name -> body)
    rules: Vec<(String, GrammarType)>,
    /// Track which primitives are needed
    needs: PrimitiveNeeds,
}

/// Tracks which primitive JSON types are needed by the grammar
#[derive(Default)]
pub struct PrimitiveNeeds {
    pub json_value: bool,
    pub json_object: bool,
    pub json_array: bool,
    pub json_kv: bool,
}

impl SchemaToGrammar {
    pub fn new() -> Self {
        Self {
            rule_counter: 0,
            rules: Vec::new(),
            needs: PrimitiveNeeds::default(),
        }
    }
    
    /// Generate a unique rule name
    fn new_rule(&mut self, prefix: &str) -> String {
        self.rule_counter += 1;
        format!("_{}{}", prefix, self.rule_counter)
    }
    
    /// Convert a SchemaType to GrammarType
    pub fn convert(&mut self, schema: &SchemaType) -> GrammarType {
        match schema {
            SchemaType::Any => {
                self.needs.json_value = true;
                GrammarType::primitive(GrammarPrimitive::JsonValue)
            }
            
            SchemaType::Never => {
                // This should never match - emit a special marker
                GrammarType::lit("<NEVER>")
            }
            
            SchemaType::Primitive(p) => self.convert_primitive(*p),
            
            SchemaType::String(constraints) => self.convert_string(constraints),
            
            SchemaType::Object(obj) => self.convert_object(obj),
            
            SchemaType::Array(arr) => self.convert_array(arr),
            
            SchemaType::Const(value) => self.convert_const(value),
            
            SchemaType::Enum(values) => self.convert_enum(values),
            
            SchemaType::Ref(path) => {
                // For now, create a rule reference
                // The actual rule should have been created earlier
                let rule_name = self.ref_path_to_rule_name(path);
                GrammarType::RuleRef(rule_name)
            }
            
            SchemaType::AllOf(schemas) => {
                // For allOf, we need to merge the schemas
                // This is complex and depends on the schema types
                // For now, just convert the first one (simplification)
                if schemas.is_empty() {
                    self.needs.json_value = true;
                    GrammarType::primitive(GrammarPrimitive::JsonValue)
                } else {
                    // Try to merge object schemas
                    self.convert_all_of(schemas)
                }
            }
            
            SchemaType::AnyOf(schemas) | SchemaType::OneOf(schemas) => {
                let alternatives: Vec<GrammarType> = schemas.iter()
                    .map(|s| self.convert(s))
                    .collect();
                GrammarType::choice(alternatives)
            }
            
            SchemaType::MultiType(schemas) => {
                let alternatives: Vec<GrammarType> = schemas.iter()
                    .map(|s| self.convert(s))
                    .collect();
                GrammarType::choice(alternatives)
            }
        }
    }
    
    /// Sanitize a name to be a valid EBNF rule name
    fn sanitize_rule_name(name: &str) -> String {
        let mut result = String::with_capacity(name.len());
        for c in name.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => result.push(c),
                '<' | '>' | '[' | ']' | '(' | ')' | '{' | '}' => result.push('_'),
                '/' | '-' | '.' | ':' | '#' | ' ' => result.push('_'),
                _ => result.push('_'),
            }
        }
        result
    }
    
    fn ref_path_to_rule_name(&self, path: &str) -> String {
        // Convert ref paths to rule names:
        // #/$defs/Foo -> _def_Foo
        // #/definitions/Bar -> _def_Bar
        // #/defs/Baz -> _def_Baz
        // #/refs/Qux -> _ref_Qux
        if let Some(name) = path.strip_prefix("#/$defs/") {
            format!("_def_{}", Self::sanitize_rule_name(name))
        } else if let Some(name) = path.strip_prefix("#/definitions/") {
            format!("_def_{}", Self::sanitize_rule_name(name))
        } else if let Some(name) = path.strip_prefix("#/defs/") {
            format!("_def_{}", Self::sanitize_rule_name(name))
        } else if let Some(name) = path.strip_prefix("#/refs/") {
            format!("_ref_{}", Self::sanitize_rule_name(name))
        } else {
            // Fallback: use the last segment
            let name = path.rsplit('/').next().unwrap_or("ref");
            format!("_ref_{}", Self::sanitize_rule_name(name))
        }
    }
    
    fn convert_primitive(&mut self, p: PrimitiveType) -> GrammarType {
        match p {
            PrimitiveType::Integer => GrammarType::primitive(GrammarPrimitive::JsonInteger),
            PrimitiveType::Number => GrammarType::primitive(GrammarPrimitive::JsonNumber),
            PrimitiveType::Boolean => GrammarType::primitive(GrammarPrimitive::JsonBoolean),
            PrimitiveType::Null => GrammarType::primitive(GrammarPrimitive::JsonNull),
        }
    }
    
    fn convert_string(&mut self, constraints: &StringConstraints) -> GrammarType {
        if constraints.is_empty() {
            // Simple string, no constraints
            return GrammarType::primitive(GrammarPrimitive::JsonString);
        }
        
        // String with constraints - build custom pattern
        let content = self.build_string_content(constraints);
        
        GrammarType::seq(vec![
            GrammarType::lit("\""),
            content,
            GrammarType::lit("\""),
        ])
    }
    
    fn build_string_content(&mut self, constraints: &StringConstraints) -> GrammarType {
        // For now, handle length constraints
        // Pattern handling would require regex-to-grammar conversion
        
        let char_or_escape = GrammarType::choice(vec![
            GrammarType::primitive(GrammarPrimitive::StringChar),
            GrammarType::primitive(GrammarPrimitive::EscapeSeq),
        ]);
        
        match (constraints.min_length, constraints.max_length) {
            (None, None) => GrammarType::primitive(GrammarPrimitive::StringChars),
            (Some(min), None) => {
                // min chars required, then unbounded: char_or_escape{min,}
                let min = min as usize;
                if min == 0 {
                    GrammarType::primitive(GrammarPrimitive::StringChars)
                } else {
                    GrammarType::seq(vec![
                        GrammarType::RepeatBounded {
                            min,
                            max: None,
                            inner: Box::new(char_or_escape),
                        }
                    ])
                }
            }
            (None, Some(max)) => {
                // 0 to max chars: char_or_escape{0,max}
                let max = max as usize;
                if max == 0 {
                    GrammarType::seq(vec![]) // empty sequence
                } else {
                    GrammarType::RepeatBounded {
                        min: 0,
                        max: Some(max),
                        inner: Box::new(char_or_escape),
                    }
                }
            }
            (Some(min), Some(max)) => {
                // min to max chars: char_or_escape{min,max}
                let min = min as usize;
                let max = max as usize;
                if max == 0 {
                    GrammarType::seq(vec![]) // empty sequence
                } else if min == max {
                    // Exact count
                    GrammarType::RepeatBounded {
                        min,
                        max: Some(max),
                        inner: Box::new(char_or_escape),
                    }
                } else {
                    GrammarType::RepeatBounded {
                        min,
                        max: Some(max),
                        inner: Box::new(char_or_escape),
                    }
                }
            }
        }
    }
    
    fn convert_object(&mut self, obj: &ObjectSchema) -> GrammarType {
        // Build the object content pattern
        //
        // Structure:
        // '{' [ property_sequence ] [ ',' additional_props ]* '}'
        //
        // Property sequence handles required vs optional properties
        
        if obj.properties.is_empty() && obj.additional_properties.is_some() {
            // Generic object
            self.needs.json_object = true;
            return GrammarType::primitive(GrammarPrimitive::JsonObject);
        }
        
        if obj.properties.is_empty() && obj.additional_properties.is_none() {
            // Empty object only
            return GrammarType::seq(vec![
                GrammarType::lit("{"),
                GrammarType::lit("}"),
            ]);
        }
        
        // Build property patterns
        let content = self.build_object_content(obj);
        
        GrammarType::JsonObject {
            open: Box::new(GrammarType::lit("{")),
            content: Box::new(content),
            close: Box::new(GrammarType::lit("}")),
        }
    }
    
    fn build_object_content(&mut self, obj: &ObjectSchema) -> GrammarType {
        // Strategy: Properties appear in order, each property is optional unless required
        // Additional properties can appear after all declared properties
        //
        // For 2 properties (a required, b optional) with additional:
        // '"a"' ':' valueA (',' '"b"' ':' valueB)? (',' _json_kv)*
        //
        // For 2 properties (both optional) with additional:
        // ('"a"' ':' valueA (',' '"b"' ':' valueB)? (',' _json_kv)*)?
        // | ('"b"' ':' valueB (',' _json_kv)*)?
        // | ((_json_kv (',' _json_kv)*)?)?
        
        let mut property_patterns: Vec<(String, GrammarType, bool)> = Vec::new();
        
        for (name, schema, required) in &obj.properties {
            let value_grammar = self.convert(schema);
            property_patterns.push((name.clone(), value_grammar, *required));
        }
        
        // Build the grammar for the properties
        let has_additional = obj.additional_properties.is_some();
        
        if property_patterns.is_empty() && has_additional {
            // Just additional properties
            self.needs.json_kv = true;
            let kv_ref = GrammarType::RuleRef("_json_kv".to_string());
            let comma_kv = GrammarType::seq(vec![GrammarType::lit(","), kv_ref.clone()]);
            return GrammarType::opt(GrammarType::seq(vec![
                kv_ref,
                GrammarType::repeat(comma_kv),
            ]));
        }
        
        // Build the property sequence
        let props_grammar = self.build_property_sequence(&property_patterns, has_additional);
        
        props_grammar
    }
    
    fn build_property_sequence(&mut self, props: &[(String, GrammarType, bool)], has_additional: bool) -> GrammarType {
        // Build a pattern that handles required/optional properties in order
        //
        // Key insight: We need to handle all valid orderings where:
        // 1. Required properties must appear
        // 2. Optional properties may appear
        // 3. Properties appear in declaration order
        
        if props.is_empty() {
            return GrammarType::Empty;
        }
        
        let additional_suffix = if has_additional {
            self.needs.json_kv = true;
            let kv_ref = GrammarType::RuleRef("_json_kv".to_string());
            let comma_kv = GrammarType::seq(vec![GrammarType::lit(","), kv_ref]);
            Some(GrammarType::repeat(comma_kv))
        } else {
            None
        };
        
        // Build property key-value patterns
        let prop_kvs: Vec<GrammarType> = props.iter()
            .map(|(name, value, _required)| {
                let escaped_key = escape_string_for_json(name);
                GrammarType::JsonKeyValue {
                    key: Box::new(GrammarType::lit(&format!("\"{}\"", escaped_key))),
                    colon: Box::new(GrammarType::lit(":")),
                    value: Box::new(value.clone()),
                }
            })
            .collect();
        
        // Check if all properties are required
        let all_required = props.iter().all(|(_, _, r)| *r);
        let all_optional = props.iter().all(|(_, _, r)| !*r);
        
        if all_required {
            // All required: prop1, prop2, prop3, additional*
            let mut parts = Vec::new();
            for (i, kv) in prop_kvs.iter().enumerate() {
                if i > 0 {
                    parts.push(GrammarType::lit(","));
                }
                parts.push(kv.clone());
            }
            if let Some(suffix) = additional_suffix {
                parts.push(suffix);
            }
            GrammarType::seq(parts)
        } else if all_optional && props.len() <= 3 {
            // All optional with few properties: enumerate all valid combinations
            self.build_optional_property_choices(&prop_kvs, additional_suffix)
        } else {
            // Mixed or many properties: use a simplified approach
            // Required props first, then optional ones with commas
            self.build_mixed_property_pattern(props, &prop_kvs, additional_suffix)
        }
    }
    
    fn build_optional_property_choices(&mut self, prop_kvs: &[GrammarType], additional_suffix: Option<GrammarType>) -> GrammarType {
        // Build all valid orderings for optional properties
        // For 2 props: empty | a | b | a,b
        // For 3 props: empty | a | b | c | a,b | a,c | b,c | a,b,c
        
        let n = prop_kvs.len();
        let mut alternatives = Vec::new();
        
        // Generate all subsets (power set)
        for mask in 0..(1 << n) {
            let mut parts = Vec::new();
            let mut first = true;
            
            for (i, kv) in prop_kvs.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    if !first {
                        parts.push(GrammarType::lit(","));
                    }
                    parts.push(kv.clone());
                    first = false;
                }
            }
            
            if let Some(ref suffix) = additional_suffix {
                if !parts.is_empty() {
                    parts.push(suffix.clone());
                } else {
                    // Empty case with additional
                    let kv_ref = GrammarType::RuleRef("_json_kv".to_string());
                    let comma_kv = GrammarType::seq(vec![GrammarType::lit(","), kv_ref.clone()]);
                    alternatives.push(GrammarType::opt(GrammarType::seq(vec![
                        kv_ref,
                        GrammarType::repeat(comma_kv),
                    ])));
                    continue;
                }
            }
            
            if parts.is_empty() {
                alternatives.push(GrammarType::Empty);
            } else {
                alternatives.push(GrammarType::seq(parts));
            }
        }
        
        GrammarType::choice(alternatives)
    }
    
    fn build_mixed_property_pattern(&mut self, props: &[(String, GrammarType, bool)], prop_kvs: &[GrammarType], additional_suffix: Option<GrammarType>) -> GrammarType {
        // Count optional properties
        let optional_count = props.iter().filter(|(_, _, r)| !*r).count();
        
        // If many optional properties, use a different strategy to avoid exponential blowup
        // The threshold of 5 is chosen because 2^5 = 32 which is still manageable,
        // but 2^22 = 4 million which is not.
        if optional_count > 0 {
            return self.build_interleaved_property_pattern(props, prop_kvs, additional_suffix);
        }
        
        // Original approach for small number of optional properties
        let mut parts = Vec::new();
        let mut first_required_seen = false;
        let mut pending_optionals: Vec<GrammarType> = Vec::new();
        
        for ((_name, _value, required), kv) in props.iter().zip(prop_kvs.iter()) {
            if *required {
                // Flush pending optionals
                for opt in pending_optionals.drain(..) {
                    parts.push(GrammarType::opt(GrammarType::seq(vec![
                        GrammarType::lit(","),
                        opt,
                    ])));
                }
                
                if first_required_seen || !parts.is_empty() {
                    parts.push(GrammarType::lit(","));
                }
                parts.push(kv.clone());
                first_required_seen = true;
            } else {
                pending_optionals.push(kv.clone());
            }
        }
        
        // Flush remaining optionals
        for opt in pending_optionals.drain(..) {
            parts.push(GrammarType::opt(GrammarType::seq(vec![
                GrammarType::lit(","),
                opt,
            ])));
        }
        
        if let Some(suffix) = additional_suffix {
            parts.push(suffix);
        }
        
        let seq = GrammarType::seq(parts);
        
        // If no required properties, make the whole thing optional
        if !first_required_seen {
            GrammarType::opt(seq)
        } else {
            seq
        }
    }
    
    /// Build property pattern using interleaved approach for many optional properties.
    /// 
    /// Instead of generating:
    ///   req1 ',' opt1? ',' opt2? ... ',' optN?
    /// which leads to 2^N variants during null inlining,
    /// 
    /// We generate:
    ///   req1 additional_opts
    ///   additional_opts ::= (',' (opt1 | opt2 | ... | optN) additional_opts)?
    ///
    /// This uses right recursion instead of sequential optionals, avoiding the exponential blowup.
    /// 
    /// For mixed cases with multiple required properties interspersed with optionals,
    /// we generate:
    ///   (opts_before_req1)? req1 (opts_after_req1)? ',' req2 (opts_after_req2)?
    /// where each opts_group uses the recursive pattern if there are many optionals.
    fn build_interleaved_property_pattern(&mut self, props: &[(String, GrammarType, bool)], prop_kvs: &[GrammarType], additional_suffix: Option<GrammarType>) -> GrammarType {
        // Separate required and optional properties
        let required_props: Vec<(usize, &GrammarType)> = props.iter().zip(prop_kvs.iter())
            .enumerate()
            .filter(|(_, ((_, _, r), _))| *r)
            .map(|(i, (_, kv))| (i, kv))
            .collect();
        
        let optional_props: Vec<&GrammarType> = props.iter().zip(prop_kvs.iter())
            .filter(|((_, _, r), _)| !*r)
            .map(|(_, kv)| kv)
            .collect();
        
        if required_props.is_empty() {
            // All optional: generate choice pattern with repetition
            let mut opt_choices: Vec<GrammarType> = optional_props.iter()
                .map(|kv| (*kv).clone())
                .collect();
            
            if let Some(ref suffix) = additional_suffix {
                // Add _json_kv as another option
                self.needs.json_kv = true;
                opt_choices.push(GrammarType::RuleRef("_json_kv".to_string()));
            }
            
            if opt_choices.is_empty() {
                return GrammarType::Empty;
            }
            
            // Generate: (prop (',' prop)*)? where prop is any optional property
            let prop_choice = GrammarType::choice(opt_choices);
            let comma_prop = GrammarType::seq(vec![GrammarType::lit(","), prop_choice.clone()]);
            
            GrammarType::opt(GrammarType::seq(vec![
                prop_choice,
                GrammarType::repeat(comma_prop),
            ]))
        } else {
            // Has required properties: build sequence with optional interleaving
            let mut parts = Vec::new();
            
            // Build the optional property choice pattern
            let mut opt_choices: Vec<GrammarType> = optional_props.iter()
                .map(|kv| (*kv).clone())
                .collect();
            
            if let Some(ref _suffix) = additional_suffix {
                self.needs.json_kv = true;
                opt_choices.push(GrammarType::RuleRef("_json_kv".to_string()));
            }
            
            let opt_repeat = if !opt_choices.is_empty() {
                let prop_choice = GrammarType::choice(opt_choices);
                let comma_prop = GrammarType::seq(vec![GrammarType::lit(","), prop_choice]);
                Some(GrammarType::repeat(comma_prop))
            } else if additional_suffix.is_some() {
                // Just additional properties
                let kv_ref = GrammarType::RuleRef("_json_kv".to_string());
                let comma_kv = GrammarType::seq(vec![GrammarType::lit(","), kv_ref]);
                Some(GrammarType::repeat(comma_kv))
            } else {
                None
            };
            
            // Add required properties with optional properties interspersed
            for (i, (_, kv)) in required_props.iter().enumerate() {
                if i > 0 {
                    // Add optional properties between required ones
                    if let Some(ref opt_rep) = opt_repeat {
                        parts.push(opt_rep.clone());
                    }
                    parts.push(GrammarType::lit(","));
                }
                parts.push((*kv).clone());
            }
            
            // Add trailing optional properties
            if let Some(opt_rep) = opt_repeat {
                parts.push(opt_rep);
            }
            
            GrammarType::seq(parts)
        }
    }
    
    fn convert_array(&mut self, arr: &ArraySchema) -> GrammarType {
        if arr.prefix_items.is_empty() && arr.items.is_none() {
            // Generic array
            self.needs.json_array = true;
            return GrammarType::primitive(GrammarPrimitive::JsonArray);
        }
        
        let content = self.build_array_content(arr);
        
        GrammarType::JsonArray {
            open: Box::new(GrammarType::lit("[")),
            content: Box::new(content),
            close: Box::new(GrammarType::lit("]")),
        }
    }
    
    fn build_array_content(&mut self, arr: &ArraySchema) -> GrammarType {
        // Handle prefix items (tuple-style)
        if !arr.prefix_items.is_empty() {
            let mut parts = Vec::new();
            
            for (i, item_schema) in arr.prefix_items.iter().enumerate() {
                if i > 0 {
                    parts.push(GrammarType::lit(","));
                }
                parts.push(self.convert(item_schema));
            }
            
            // Handle additional items after prefix
            if let Some(additional) = &arr.additional_items {
                let item_grammar = self.convert(additional);
                let comma_item = GrammarType::seq(vec![GrammarType::lit(","), item_grammar]);
                parts.push(GrammarType::repeat(comma_item));
            } else if let Some(items) = &arr.items {
                let item_grammar = self.convert(items);
                let comma_item = GrammarType::seq(vec![GrammarType::lit(","), item_grammar]);
                parts.push(GrammarType::repeat(comma_item));
            }
            
            // Make the whole thing optional if empty array is allowed
            GrammarType::opt(GrammarType::seq(parts))
        } else if let Some(items) = &arr.items {
            // Homogeneous array
            let item_grammar = self.convert(items);
            let comma_item = GrammarType::seq(vec![GrammarType::lit(","), item_grammar.clone()]);
            
            GrammarType::opt(GrammarType::seq(vec![
                item_grammar,
                GrammarType::repeat(comma_item),
            ]))
        } else {
            GrammarType::Empty
        }
    }
    
    fn convert_const(&mut self, value: &Value) -> GrammarType {
        self.value_to_grammar(value)
    }
    
    fn convert_enum(&mut self, values: &[Value]) -> GrammarType {
        let alternatives: Vec<GrammarType> = values.iter()
            .map(|v| self.value_to_grammar(v))
            .collect();
        GrammarType::choice(alternatives)
    }
    
    fn value_to_grammar(&mut self, value: &Value) -> GrammarType {
        match value {
            Value::Null => GrammarType::lit("null"),
            Value::Bool(true) => GrammarType::lit("true"),
            Value::Bool(false) => GrammarType::lit("false"),
            Value::Number(n) => GrammarType::Literal(n.to_string().into_bytes()),
            Value::String(s) => {
                let escaped = escape_string_for_json(s);
                GrammarType::Literal(format!("\"{}\"", escaped).into_bytes())
            }
            Value::Array(items) => {
                let mut parts = vec![GrammarType::lit("[")];
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        parts.push(GrammarType::lit(","));
                    }
                    parts.push(self.value_to_grammar(item));
                }
                parts.push(GrammarType::lit("]"));
                GrammarType::seq(parts)
            }
            Value::Object(map) => {
                let mut parts = vec![GrammarType::lit("{")];
                for (i, (key, val)) in map.iter().enumerate() {
                    if i > 0 {
                        parts.push(GrammarType::lit(","));
                    }
                    let escaped_key = escape_string_for_json(key);
                    parts.push(GrammarType::Literal(format!("\"{}\"", escaped_key).into_bytes()));
                    parts.push(GrammarType::lit(":"));
                    parts.push(self.value_to_grammar(val));
                }
                parts.push(GrammarType::lit("}"));
                GrammarType::seq(parts)
            }
        }
    }
    
    fn convert_all_of(&mut self, schemas: &[SchemaType]) -> GrammarType {
        // allOf is tricky - we need to merge the schemas
        // For simplicity, if they're all objects, merge properties
        // Otherwise, just use the first one
        
        let mut merged_props: Vec<(String, GrammarType, bool)> = Vec::new();
        let mut additional = Some(Box::new(SchemaType::Any));
        
        for schema in schemas {
            if let SchemaType::Object(obj) = schema {
                for (name, prop_schema, required) in &obj.properties {
                    // Check if property already exists
                    if let Some(pos) = merged_props.iter().position(|(n, _, _)| n == name) {
                        // Update required status (if any says required, it's required)
                        merged_props[pos].2 = merged_props[pos].2 || *required;
                    } else {
                        merged_props.push((name.clone(), self.convert(prop_schema), *required));
                    }
                }
                
                // Merge additionalProperties (most restrictive wins)
                if obj.additional_properties.is_none() {
                    additional = None;
                }
            } else {
                // Non-object schema in allOf - just return first schema
                return self.convert(&schemas[0]);
            }
        }
        
        // Build merged object
        let obj_schema = ObjectSchema {
            properties: merged_props.iter()
                .map(|(n, _, r)| (n.clone(), SchemaType::Any, *r))
                .collect(),
            additional_properties: additional,
            pattern_properties: vec![],
        };
        
        // Build directly using the property patterns we already have
        let content = self.build_property_sequence(
            &merged_props.iter().map(|(n, g, r)| (n.clone(), g.clone(), *r)).collect::<Vec<_>>(),
            obj_schema.additional_properties.is_some()
        );
        
        GrammarType::JsonObject {
            open: Box::new(GrammarType::lit("{")),
            content: Box::new(content),
            close: Box::new(GrammarType::lit("}")),
        }
    }
    
    /// Get the generated rules
    pub fn get_rules(&self) -> &[(String, GrammarType)] {
        &self.rules
    }
    
    /// Get the primitive needs
    pub fn get_needs(&self) -> &PrimitiveNeeds {
        &self.needs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_convert_primitive() {
        let mut conv = SchemaToGrammar::new();
        let g = conv.convert(&SchemaType::Primitive(PrimitiveType::Integer));
        assert!(matches!(g, GrammarType::PrimitiveRef(GrammarPrimitive::JsonInteger)));
    }
    
    #[test]
    fn test_convert_simple_object() {
        let mut conv = SchemaToGrammar::new();
        let schema = SchemaType::Object(ObjectSchema {
            properties: vec![
                ("name".to_string(), SchemaType::String(StringConstraints::default()), true),
            ],
            additional_properties: None,
            pattern_properties: vec![],
        });
        let g = conv.convert(&schema);
        assert!(matches!(g, GrammarType::JsonObject { .. }));
    }
    
    #[test]
    fn test_convert_enum() {
        let mut conv = SchemaToGrammar::new();
        let schema = SchemaType::Enum(vec![
            Value::String("a".to_string()),
            Value::String("b".to_string()),
        ]);
        let g = conv.convert(&schema);
        assert!(matches!(g, GrammarType::Choice(_)));
    }
}
