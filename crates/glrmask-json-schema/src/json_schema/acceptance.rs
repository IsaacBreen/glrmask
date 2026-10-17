//! Assertion policy runs before optimizations, including in negative contexts.
//! Only schema positions are visited; instance/annotation payloads are opaque.
use serde_json::{Map, Value, json};
use super::error::{ImportResult, SchemaImportError};

pub fn prepare_document(root: &Value) -> ImportResult<Value> {
    prepare(root, "#")
}

fn unsupported(path: &str, feature: &str) -> SchemaImportError {
    SchemaImportError::at(path, format!("unsupported assertion: {feature}"))
}
fn escaped(key: &str) -> String { key.replace('~', "~0").replace('/', "~1") }

fn prepare(schema: &Value, path: &str) -> ImportResult<Value> {
    let Some(input) = schema.as_object() else {
        return if schema.is_boolean() { Ok(schema.clone()) }
        else { Err(SchemaImportError::at(path, "malformed schema: expected object or boolean")) };
    };
    let mut out = input.clone();
    // Validate before simplification, so unsupported assertions never become
    // `false` (especially under not/if/unevaluated*).
    for (key, value) in input {
        let child_path = format!("{path}/{}", escaped(key));
        match key.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
                let map = value.as_object().ok_or_else(|| SchemaImportError::at(&child_path, "malformed schema map"))?;
                let children = map.iter().map(|(name, child)| {
                    Ok((name.clone(), prepare(child, &format!("{child_path}/{}", escaped(name)))?))
                }).collect::<ImportResult<Map<String, Value>>>()?;
                out.insert(key.clone(), Value::Object(children));
            }
            "dependencies" => {
                let map = value.as_object().ok_or_else(|| SchemaImportError::at(&child_path, "malformed dependencies"))?;
                let children = map.iter().map(|(name, child)| {
                    Ok((name.clone(), if child.is_array() {child.clone()} else {prepare(child, &format!("{child_path}/{}", escaped(name)))?}))
                }).collect::<ImportResult<Map<String, Value>>>()?;
                out.insert(key.clone(), Value::Object(children));
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                let items = value.as_array().ok_or_else(|| SchemaImportError::at(&child_path, "malformed schema array"))?;
                if items.is_empty() { return Err(SchemaImportError::at(&child_path, "malformed schema: schema array must be nonempty")); }
                let children = items.iter().enumerate().map(|(i, child)| prepare(child, &format!("{child_path}/{i}"))).collect::<ImportResult<Vec<_>>>()?;
                out.insert(key.clone(), Value::Array(children));
            }
            "items" if value.is_array() => {
                let children = value.as_array().unwrap().iter().enumerate().map(|(i, child)| prepare(child, &format!("{child_path}/{i}"))).collect::<ImportResult<Vec<_>>>()?;
                out.insert(key.clone(), Value::Array(children));
            }
            "items" | "additionalItems" | "additionalProperties" | "propertyNames" | "not" | "if" | "then" | "else" | "contains" | "unevaluatedProperties" | "unevaluatedItems" => {
                out.insert(key.clone(), prepare(value, &child_path)?);
            }
            _ => {}
        }
    }
    if let Some(unique) = out.remove("uniqueItems") {
        let unique = unique.as_bool().ok_or_else(|| SchemaImportError::at(&format!("{path}/uniqueItems"), "malformed uniqueItems: expected boolean"))?;
        let no_arrays = out.get("type").is_some_and(|t| match t {
            Value::String(t) => t != "array",
            Value::Array(ts) => ts.iter().all(|t| t.as_str().is_some_and(|t| t != "array")),
            _ => false,
        });
        let at_most_one = out.get("maxItems").and_then(Value::as_u64).is_some_and(|m| m <= 1)
            || (out.get("items") == Some(&Value::Bool(false))
                && out.get("prefixItems").and_then(Value::as_array).is_none_or(|p| p.len() <= 1));
        if unique && !no_arrays && !at_most_one {
            return Err(unsupported(&format!("{path}/uniqueItems"), "uniqueItems:true requires arrays of at most one item"));
        }
    }
    if let Some(negated) = out.remove("not") {
        let complement = if negated == Value::Bool(false) || negated == json!({"not":true}) { Value::Bool(true) }
        else if negated == Value::Bool(true) || negated == json!({}) { Value::Bool(false) }
        else if let Some(map) = negated.as_object().filter(|m| m.len() == 1 && m.contains_key("type")) {
            let types = match &map["type"] {
                Value::String(t) => vec![t.as_str()],
                Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
                _ => return Err(unsupported(&format!("{path}/not"), "negation of malformed type")),
            };
            if types.contains(&"integer") || types.iter().any(|t| !["null","boolean","object","array","string","number"].contains(t)) {
                return Err(unsupported(&format!("{path}/not"), "only boolean and primitive type complements are supported"));
            }
            let remaining: Vec<_> = ["null","boolean","object","array","string","number"].into_iter().filter(|t| !types.contains(t)).collect();
            if remaining.is_empty() { Value::Bool(false) } else { json!({"type":remaining}) }
        } else {
            return Err(unsupported(&format!("{path}/not"), "general negation; only boolean and primitive type complements are supported"));
        };
        append_all_of(&mut out, complement);
    }
    let test = out.remove("if");
    let then = out.remove("then");
    let otherwise = out.remove("else");
    if let Some(test) = test {
        let selected = if test == Value::Bool(true) || test == json!({}) { then }
        else if test == Value::Bool(false) { otherwise }
        else if then.as_ref().is_none_or(is_true) && otherwise.as_ref().is_none_or(is_true) { None }
        else { return Err(unsupported(&format!("{path}/if"), "conditional assertions require a constant condition or vacuous branches")); };
        if let Some(selected) = selected { append_all_of(&mut out, selected); }
    }
    // Current combinator sibling extraction cannot preserve two independent
    // combinators. Refuse the combination before either assertion is lost.
    if ["allOf", "anyOf", "oneOf"].iter().filter(|key| out.contains_key(**key)).count() > 1 {
        return Err(unsupported(path, "multiple sibling combinators require an exact intersection"));
    }
    for key in ["unevaluatedProperties", "unevaluatedItems"] {
        if out.get(key) == Some(&Value::Bool(true)) { out.remove(key); }
    }
    if !out.contains_key("contains") { out.remove("minContains"); out.remove("maxContains"); }
    if out.get("dependentSchemas") == Some(&json!({})) { out.remove("dependentSchemas"); }
    // Orphan then/else have no validation effect.
    Ok(Value::Object(out))
}
fn is_true(value: &Value) -> bool { value == &Value::Bool(true) || value == &json!({}) }
fn append_all_of(out: &mut Map<String, Value>, constraint: Value) {
    if is_true(&constraint) { return; }
    out.entry("allOf").or_insert_with(|| json!([])).as_array_mut().expect("validated allOf").push(constraint);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_remains_error_in_negative_and_dead_contexts() {
        for schema in [json!({"not":{"uniqueItems":true}}),json!({"allOf":[false,{"if":{"type":"string"},"then":false}]}),json!({"properties":{"a/b":{"uniqueItems":true}}})] {
            let err = prepare_document(&schema).unwrap_err();
            assert!(err.message().contains("unsupported assertion"), "{err:?}");
        }
        assert!(prepare_document(&json!({"properties":{"a/b":{"uniqueItems":true}}})).unwrap_err().message().contains("#/properties/a~1b/uniqueItems"));
    }
    #[test]
    fn exact_redundant_uniqueness_and_constant_conditionals() {
        for schema in [json!({"uniqueItems":false}),json!({"uniqueItems":true,"maxItems":1}),json!({"uniqueItems":true,"type":"string"}),json!({"if":true,"then":{"type":"integer"}}),json!({"not":{"type":"string"}})] {
            assert!(prepare_document(&schema).is_ok());
        }
        assert!(prepare_document(&json!({"uniqueItems":"true"})).unwrap_err().message().contains("malformed"));
    }
    #[test]
    fn metadata_and_instance_payloads_are_opaque() {
        let schema = json!({"default":{"uniqueItems":true},"const":{"if":{},"not":{}},"examples":[{"not":{"uniqueItems":true}}]});
        assert_eq!(prepare_document(&schema).unwrap(), schema);
    }
}
