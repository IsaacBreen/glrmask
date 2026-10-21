use glrmask::{BuildOptions, Grammar, MaskStrategy, Vocab};
use serde_json::{Value, json};

fn check(schema: Value, examples: &[(&str, bool)]) {
    let vocab = Vocab::new((0..=255).map(|b| (b, vec![b as u8])).collect());
    let text = schema.to_string();
    for mode in [MaskStrategy::TokenTrie, MaskStrategy::ParserDwa] {
        let constraint = Grammar::from_json_schema(&text).compile_with_build_options(&vocab, BuildOptions::default().optimization(mode)).unwrap_or_else(|e| panic!("{schema} {mode:?}: {e}"));
        for (text, expected) in examples {
            let mut state = constraint.start();
            let mut possible = true;
            for byte in text.bytes() {
                let mask = state.mask();
                if mask.get(byte as usize/32).is_none_or(|m| m & (1 << (byte as usize%32)) == 0) { possible = false; break; }
                state.commit_token(u32::from(byte)).unwrap();
            }
            assert_eq!(possible && state.is_accepting(), *expected, "{schema} {text} {mode:?}");
        }
    }
}
fn refuses(schema: Value, location: &str) {
    let vocab = Vocab::new(vec![(0,b"x".to_vec())]);
    let text=schema.to_string();
    for mode in [MaskStrategy::TokenTrie, MaskStrategy::ParserDwa] {
        let error = Grammar::from_json_schema(&text).compile_with_build_options(&vocab,BuildOptions::default().optimization(mode)).err().expect("must fail closed").to_string();
        assert!(error.contains(location), "{error}");
        assert!(error.contains("unsupported") || error.contains("Unimplemented"), "{error}");
    }
}
#[test]
fn exact_oneof_and_restricted_assertions_enforce_instances() {
    check(json!({"oneOf":[{"type":"string"},{"type":"integer"}]}), &[("1",true),("\"s\"",true),("false",false)]);
    check(json!({"type":"array","maxItems":1,"uniqueItems":true,"items":{"type":"integer"}}), &[("[]",true),("[1]",true),("[1, 1]",false)]);
    check(json!({"not":{"type":"string"}}), &[("1",true),("false",true),("\"s\"",false)]);
    check(json!({"if":true,"then":{"type":"integer","minimum":2},"else":false}), &[("1",false),("2",true),("\"s\"",false)]);
}
#[test]
fn unsupported_assertions_keep_paths_in_nested_contexts() {
    refuses(json!({"properties":{"x":{"uniqueItems":true}}}), "#/properties/x/uniqueItems");
    refuses(json!({"not":{"uniqueItems":true}}), "#/not/uniqueItems");
    refuses(json!({"oneOf":[{"type":"integer"},{"type":"number"}]}), "#/oneOf/1");
    refuses(json!({"if":{"type":"integer"},"then":false}), "#/if");
    refuses(json!({"unevaluatedProperties":false}), "#");
    refuses(json!({"type":"object","properties":{"a":true},"minProperties":1}), "#");
}
#[test]
fn annotations_and_format_vocabularies_are_distinct() {
    check(json!({"type":"string","format":"email","default":{"uniqueItems":true},"x-extension":{"not":true}}), &[("\"anything\"",true),("1",false)]);
    refuses(json!({"$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/format-assertion":true},"format":"email"}), "#/$vocabulary");
    refuses(json!({"$vocabulary":{"https://example.invalid/required":true}}), "#/$vocabulary");
    check(json!({"$vocabulary":{"https://example.invalid/optional":false},"type":"boolean"}), &[("true",true),("1",false)]);
}
#[test]
fn pattern_and_finite_assertions_intersect_without_dropping_length() {
    check(json!({"enum":["a","aa"],"pattern":"a","minLength":2}), &[("\"a\"",false),("\"aa\"",true)]);
    check(json!({"const":1,"enum":[1.0]}), &[("1",true),("2",false)]);
    check(json!({"type":"object","additionalProperties":false,"properties":{"a":true,"b":true},"propertyNames":{"pattern":"^a$"}}), &[("{}",true),("{\"a\": 1}",true),("{\"b\": 1}",false)]);
}
#[test]
fn older_ref_siblings_follow_declared_dialect() {
    check(json!({"$schema":"http://json-schema.org/draft-07/schema#","definitions":{"x":{"type":"integer"}},"$ref":"#/definitions/x","type":"string"}), &[("1",true),("\"s\"",false)]);
    check(json!({"$defs":{"x":{"type":"integer"}},"$ref":"#/$defs/x","minimum":2}), &[("1",false),("2",true)]);
}

#[test]
fn object_union_merges_and_dynamic_minima_cannot_broaden_assertions() {
    refuses(json!({"anyOf":[
        {"type":"object","allOf":[
            {"properties":{"a":true},"additionalProperties":false},
            {"properties":{"b":true},"additionalProperties":false}
        ]},
        {"type":"object","properties":{"c":true},"additionalProperties":false}
    ]}), "#/anyOf/0/allOf");
    refuses(json!({"anyOf":[
        {"allOf":[
            {"type":"object","propertyNames":{"pattern":"^a$"}},
            {"type":"object","propertyNames":{"pattern":"^b$"}}
        ]},
        {"type":"object","additionalProperties":false}
    ]}), "#/anyOf/0/allOf");
    // Repeating a dynamic key would satisfy a textual member count of two but
    // leave only one property after ordinary JSON parsing. Refuse that shape.
    refuses(json!({"minProperties":2}), "#");
    refuses(json!({"type":"object","minProperties":2,"patternProperties":{"^a":true},"additionalProperties":false}), "#");
    check(json!({"type":"object","minProperties":1}), &[
        ("{}",false), ("{\"a\": 1}",true), ("{\"a\": 1, \"a\": 2}",true)
    ]);
    // Distinct required fixed names and finite literals can still prove the
    // bound without imposing uniqueness on an unrestricted key language.
    check(json!({"type":"object","minProperties":2,"required":["a","b"]}), &[
        ("{\"a\": 1, \"b\": 2}",true), ("{\"a\": 1, \"a\": 2}",false)
    ]);
    check(json!({"const":{"a":1,"b":2},"minProperties":2}), &[
        ("{\"a\": 1, \"b\": 2}",true), ("{\"a\": 1}",false)
    ]);
}

#[test]
fn allof_identity_and_object_type_guards_are_preserved() {
    for schema in [
        json!({"allOf":[{"type":"object"},true]}),
        json!({"allOf":[true,{"type":"object"}]}),
        json!({"type":"object","allOf":[true]}),
        json!({"allOf":[{"type":"object"},{},true]})
    ] { check(schema, &[("{}",true),("1",false),("\"s\"",false),("true",false)]); }
    check(json!({"allOf":[true,true]}), &[("{}",true),("1",true)]);
    refuses(json!({"allOf":[{"type":"object"},{"$ref":"#/$defs/r"}],"$defs":{"r":{"anyOf":[{"$ref":"#/$defs/r"},{"type":"string"}]}}}), "#");
}

#[test]
fn allof_preserves_untyped_and_union_string_alternatives() {
    check(json!({"allOf":[{"minLength":2}]}), &[("5",true),("true",true),("\"ab\"",true),("\"a\"",false)]);
    check(json!({"allOf":[{"type":["string","boolean"],"minLength":2}]}), &[("true",true),("false",true),("\"ab\"",true),("\"a\"",false),("5",false)]);
    check(json!({"allOf":[{"type":["string","null"],"minLength":2}]}), &[("null",true),("\"ab\"",true),("\"a\"",false)]);
    // Multiple untyped operands require a parser-language intersection. An
    // explicit refusal is preferable to discarding every nonstring value.
    refuses(json!({"allOf":[{"minLength":2},{"maxLength":3}]}), "#");
}

#[test]
fn nested_allof_boolean_identities_preserve_type_intersections() {
    for schema in [
        json!({"allOf":[{"type":"object"},{"allOf":[true,true]}]}),
        json!({"allOf":[true,{"allOf":[{"type":"object"},true]}]}),
        json!({"allOf":[{"type":"object"},{"anyOf":[true]},{"anyOf":[true]}]})
    ] { check(schema, &[("{}",true),("1",false),("\"s\"",false)]); }
    for schema in [
        json!({"allOf":[{"type":"object"},{"allOf":[true,false]}]}),
        json!({"allOf":[{"allOf":[{"type":"object"},true]},{"allOf":[true,false]}]}),
        json!({"allOf":[{"allOf":[{"type":"object"},true]},{"type":"string"}]}),
        json!({"allOf":[{"type":"object"},{"allOf":[{"type":"array"},true]}]})
    ] { check(schema, &[("{}",false),("[]",false),("1",false),("\"s\"",false)]); }
    for schema in [
        json!({"allOf":[{"allOf":[{"type":"integer"},true]},{"type":"number"}]}),
        json!({"allOf":[{"allOf":[{"type":"number"},true]},{"type":"integer"}]})
    ] { check(schema, &[("1",true),("1.5",false),("\"s\"",false)]); }
    check(json!({"allOf":[{"type":["object","string"]},{"allOf":[{"type":"string"},true]}]}), &[("\"s\"",true),("{}",false),("1",false)]);
}

#[test]
fn allof_string_guards_survive_untyped_and_union_families() {
    check(json!({"allOf":[{"type":"string"},{"minLength":2}]}), &[("\"ab\"",true),("\"a\"",false),("5",false),("{}",false)]);
    check(json!({"allOf":[{"type":"string"},{"type":["object","string"]}]}), &[("\"s\"",true),("{}",false)]);
}
