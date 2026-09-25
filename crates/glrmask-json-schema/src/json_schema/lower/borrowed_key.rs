//! Read-only view of the existing canonical memo key. This deliberately keeps
//! its binary fingerprint and full typed equality, rather than adding a new
//! schema equivalence relation or changing cache publication order.
use super::{JsonTerminalPartitionClass, StructuralSchemaCacheKey, StructuralSchemaSite};
use super::super::ast::*;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::hash::Hasher;
use rustc_hash::FxHasher;
use serde::{Serialize, Serializer};
use serde::ser::SerializeSeq;

pub(super) struct BorrowedKey<'a> {
    pub schema: &'a Schema,
    pub terminal_partition_class: JsonTerminalPartitionClass,
    pub site: StructuralSchemaSite,
    pub object_variant_ref_stack: &'a BTreeSet<String>,
}

impl BorrowedKey<'_> {
    pub fn fingerprint(&self) -> u64 {
        let encoded = bincode::serialize(self)
            .expect("loaded JSON Schema AST must remain binary-serializable");
        let mut hash = FxHasher::default();
        hash.write(&encoded);
        hash.finish()
    }

    pub fn matches(&self, key: &StructuralSchemaCacheKey) -> bool {
        key.terminal_partition_class == self.terminal_partition_class
            && key.site == self.site
            && key.object_variant_ref_stack.len() == self.object_variant_ref_stack.len()
            && key.object_variant_ref_stack.iter().eq(self.object_variant_ref_stack.iter())
            && schema_eq(&key.schema, self.schema, &self.schema.location)
    }
}

impl Serialize for BorrowedKey<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Fields<'a> {
            schema: SchemaView<'a>,
            terminal_partition_class: JsonTerminalPartitionClass,
            site: StructuralSchemaSite,
            object_variant_ref_stack: &'a BTreeSet<String>,
        }
        Fields {
            schema: SchemaView(self.schema, &self.schema.location),
            terminal_partition_class: self.terminal_partition_class,
            site: self.site,
            object_variant_ref_stack: self.object_variant_ref_stack,
        }.serialize(serializer)
    }
}

// Keep exactly Schema::normalize_locations_relative's boundary rule. Unrelated
// and synthetic locations remain unchanged; root-prefix lookalikes do too.
fn normalized_location<'a>(location: &'a str, root: &str) -> Cow<'a, str> {
    if location == root {
        Cow::Borrowed("#")
    } else if root != "#"
        && let Some(suffix) = location.strip_prefix(root)
        && suffix.starts_with('/')
    {
        Cow::Owned(format!("#{suffix}"))
    } else {
        Cow::Borrowed(location)
    }
}

fn location_eq(canonical: &str, original: &str, root: &str) -> bool {
    if original == root {
        canonical == "#"
    } else if let Some(suffix) = original.strip_prefix(root)
        && suffix.starts_with('/')
    {
        canonical.strip_prefix('#') == Some(suffix)
    } else {
        canonical == original
    }
}

#[derive(Clone, Copy)]
struct SchemaView<'a>(&'a Schema, &'a str);
impl Serialize for SchemaView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Exhaustive destructuring intentionally makes AST additions a compile
        // error here until the canonical view is updated.
        let Schema { location, kind } = self.0;
        #[derive(Serialize)]
        enum Kind<'a> {
            Any,
            Never,
            Ref(&'a str),
            Assertions(AssertionsView<'a>),
        }
        #[derive(Serialize)]
        struct Fields<'a> { location: Cow<'a, str>, kind: Kind<'a> }
        let kind = match kind {
            SchemaKind::Any => Kind::Any,
            SchemaKind::Never => Kind::Never,
            SchemaKind::Ref(pointer) => Kind::Ref(pointer),
            SchemaKind::Assertions(value) => Kind::Assertions(AssertionsView(value, self.1)),
        };
        Fields { location: normalized_location(location, self.1), kind }.serialize(serializer)
    }
}

struct Schemas<'a>(&'a [Schema], &'a str);
impl Serialize for Schemas<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for schema in self.0 { seq.serialize_element(&SchemaView(schema, self.1))?; }
        seq.end()
    }
}

struct AssertionsView<'a>(&'a SchemaAssertions, &'a str);
impl Serialize for AssertionsView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let SchemaAssertions { types, const_value, enum_values, object, array,
            string, number, any_of, one_of, all_of, not } = self.0;
        #[derive(Serialize)]
        struct Fields<'a> {
            types: &'a Option<Vec<SchemaType>>,
            const_value: &'a Option<serde_json::Value>,
            enum_values: &'a Option<Vec<serde_json::Value>>,
            object: Option<ObjectView<'a>>,
            array: Option<ArrayView<'a>>,
            string: &'a Option<StringSchema>,
            number: &'a Option<NumberSchema>,
            any_of: Schemas<'a>,
            one_of: Schemas<'a>,
            all_of: Schemas<'a>,
            not: Option<SchemaView<'a>>,
        }
        Fields {
            types, const_value, enum_values,
            object: object.as_ref().map(|v| ObjectView(v, self.1)),
            array: array.as_ref().map(|v| ArrayView(v, self.1)),
            string, number,
            any_of: Schemas(any_of, self.1), one_of: Schemas(one_of, self.1),
            all_of: Schemas(all_of, self.1), not: not.as_ref().map(|v| SchemaView(v, self.1)),
        }.serialize(serializer)
    }
}

struct Properties<'a>(&'a [PropertySchema], &'a str);
impl Serialize for Properties<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Fields<'a> { name: &'a str, schema: SchemaView<'a> }
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for PropertySchema { name, schema } in self.0 {
            seq.serialize_element(&Fields { name, schema: SchemaView(schema, self.1) })?;
        }
        seq.end()
    }
}

struct PatternProperties<'a>(&'a [PatternPropertySchema], &'a str);
impl Serialize for PatternProperties<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Fields<'a> { pattern: &'a str, schema: SchemaView<'a> }
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for PatternPropertySchema { pattern, schema } in self.0 {
            seq.serialize_element(&Fields { pattern, schema: SchemaView(schema, self.1) })?;
        }
        seq.end()
    }
}

struct ObjectView<'a>(&'a ObjectSchema, &'a str);
impl Serialize for ObjectView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let ObjectSchema { properties, required, required_order, property_dependencies,
            min_properties, max_properties, pattern_properties, property_names,
            additional_properties } = self.0;
        #[derive(Serialize)]
        enum Additional<'a> { AllowAny, Deny, Schema(SchemaView<'a>) }
        #[derive(Serialize)]
        struct Fields<'a> {
            properties: Properties<'a>,
            required: &'a BTreeSet<String>,
            required_order: &'a Vec<String>,
            property_dependencies: &'a std::collections::BTreeMap<String, BTreeSet<String>>,
            min_properties: usize,
            max_properties: Option<usize>,
            pattern_properties: PatternProperties<'a>,
            property_names: Option<SchemaView<'a>>,
            additional_properties: Additional<'a>,
        }
        let additional_properties = match additional_properties {
            AdditionalProperties::AllowAny => Additional::AllowAny,
            AdditionalProperties::Deny => Additional::Deny,
            AdditionalProperties::Schema(s) => Additional::Schema(SchemaView(s, self.1)),
        };
        Fields {
            properties: Properties(properties, self.1), required, required_order,
            property_dependencies, min_properties: *min_properties, max_properties: *max_properties,
            pattern_properties: PatternProperties(pattern_properties, self.1),
            property_names: property_names.as_ref().map(|v| SchemaView(v, self.1)),
            additional_properties,
        }.serialize(serializer)
    }
}

struct ArrayView<'a>(&'a ArraySchema, &'a str);
impl Serialize for ArrayView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let ArraySchema { items, prefix_items, min_items, max_items } = self.0;
        #[derive(Serialize)]
        struct Fields<'a> {
            items: SchemaView<'a>, prefix_items: Schemas<'a>, min_items: usize, max_items: Option<usize>,
        }
        Fields { items: SchemaView(items, self.1), prefix_items: Schemas(prefix_items, self.1),
            min_items: *min_items, max_items: *max_items }.serialize(serializer)
    }
}

fn option_eq<T>(left: &Option<T>, right: &Option<T>, eq: impl Fn(&T, &T) -> bool) -> bool {
    match (left, right) { (Some(a), Some(b)) => eq(a, b), (None, None) => true, _ => false }
}
fn schemas_eq(left: &[Schema], right: &[Schema], root: &str) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a,b)| schema_eq(a,b,root))
}
fn schema_eq(left: &Schema, right: &Schema, root: &str) -> bool {
    let Schema { location, kind } = right;
    location_eq(&left.location, location, root) && match (&left.kind, kind) {
        (SchemaKind::Any, SchemaKind::Any) | (SchemaKind::Never, SchemaKind::Never) => true,
        (SchemaKind::Ref(a), SchemaKind::Ref(b)) => a == b,
        (SchemaKind::Assertions(a), SchemaKind::Assertions(b)) => assertions_eq(a,b,root),
        _ => false,
    }
}
fn assertions_eq(left: &SchemaAssertions, right: &SchemaAssertions, root: &str) -> bool {
    let SchemaAssertions { types, const_value, enum_values, object, array,
        string, number, any_of, one_of, all_of, not } = right;
    left.types == *types && left.const_value == *const_value && left.enum_values == *enum_values
        && left.string == *string && left.number == *number
        && option_eq(&left.object, object, |a,b| object_eq(a,b,root))
        && option_eq(&left.array, array, |a,b| array_eq(a,b,root))
        && schemas_eq(&left.any_of,any_of,root) && schemas_eq(&left.one_of,one_of,root)
        && schemas_eq(&left.all_of,all_of,root)
        && option_eq(&left.not,not,|a,b| schema_eq(a,b,root))
}
fn object_eq(left: &ObjectSchema, right: &ObjectSchema, root: &str) -> bool {
    let ObjectSchema { properties, required, required_order, property_dependencies,
        min_properties, max_properties, pattern_properties, property_names, additional_properties } = right;
    left.required == *required && left.required_order == *required_order
        && left.property_dependencies == *property_dependencies
        && left.min_properties == *min_properties && left.max_properties == *max_properties
        && left.properties.len() == properties.len()
        && left.properties.iter().zip(properties).all(|(a, PropertySchema { name, schema })|
            a.name == *name && schema_eq(&a.schema,schema,root))
        && left.pattern_properties.len() == pattern_properties.len()
        && left.pattern_properties.iter().zip(pattern_properties).all(|(a, PatternPropertySchema { pattern, schema })|
            a.pattern == *pattern && schema_eq(&a.schema,schema,root))
        && option_eq(&left.property_names,property_names,|a,b| schema_eq(a,b,root))
        && match (&left.additional_properties,additional_properties) {
            (AdditionalProperties::AllowAny,AdditionalProperties::AllowAny)
            | (AdditionalProperties::Deny,AdditionalProperties::Deny) => true,
            (AdditionalProperties::Schema(a),AdditionalProperties::Schema(b)) => schema_eq(a,b,root),
            _ => false,
        }
}
fn array_eq(left: &ArraySchema, right: &ArraySchema, root: &str) -> bool {
    let ArraySchema { items,prefix_items,min_items,max_items } = right;
    left.min_items == *min_items && left.max_items == *max_items
        && schema_eq(&left.items,items,root) && schemas_eq(&left.prefix_items,prefix_items,root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_schema::load::load_document;
    use serde_json::json;

    fn owned(query: &BorrowedKey<'_>) -> StructuralSchemaCacheKey {
        let mut schema = query.schema.clone(); schema.normalize_locations_relative();
        StructuralSchemaCacheKey { schema, terminal_partition_class: query.terminal_partition_class,
            site: query.site, object_variant_ref_stack: query.object_variant_ref_stack.iter().cloned().collect() }
    }

    #[test]
    fn location_view_matches_canonical_boundary_rules() {
        for root in ["#", "#/a", "#/a/b", "", "<synthetic>", "#/한글"] {
            for location in [root.to_owned(), format!("{root}/x"),format!("{root}lookalike"),
                "#/unrelated".into(), "<implicit-array-items>".into(), format!("{root}/한글/日本語")] {
                let mut parent = Schema::assertions(root, SchemaAssertions {
                    any_of: vec![Schema::any(&location)], ..Default::default()
                });
                parent.normalize_locations_relative();
                let SchemaKind::Assertions(a) = parent.kind else { unreachable!() };
                assert_eq!(normalized_location(&location,root),a.any_of[0].location);
                assert!(location_eq(&a.any_of[0].location,&location,root));
                assert!(!location_eq("deliberately-wrong-location",&location,root));
            }
        }
    }

    #[test]
    fn every_typed_field_preserves_binary_fingerprint_and_equality() {
        let values = [
            json!(true),json!(false),json!({"$defs":{"node":{"type":"object"}},"$ref":"#/$defs/node"}),json!({}),
            json!({"type":["string","null"],"minLength":2,"maxLength":12,"pattern":"^a","format":"email"}),
            json!({"type":"number","minimum":-0.0,"maximum":42.5,"exclusiveMinimum":true,"multipleOf":0.5}),
            json!({"type":"object","properties":{"a":{"type":"string"},"한글":{"const":"日本語"}},
                "required":["a"],"minProperties":1,"maxProperties":9,"additionalProperties":{"type":"boolean"},
                "patternProperties":{"^x":{"type":"integer"}},"propertyNames":{"pattern":"a"},
                "dependencies":{"a":["한글"]}}),
            json!({"type":"array","items":{"type":"string"},"minItems":1,"maxItems":4}),
            json!({"type":"array","items":[{"type":"string"},{"type":"integer"}],"additionalItems":false}),
            json!({"anyOf":[{"const":"a"},{"const":null}]}),
            json!({"allOf":[{"type":"string"},{"maxLength":3}]}),
            json!({"oneOf":[{"type":"string"},{"type":"integer"}]}),
            json!({"enum":[null,true,false,12,12.5,"日本語",[1,"x"],{"z":1,"a":2}]}),
            json!({"not":{"type":"object"}}),
        ];
        let mut schemas: Vec<_> = values.iter().map(|v| load_document(v).unwrap().root).collect();
        // Also exercise optional fields and all recursive children together,
        // independent of which combinations the loader simplifies away.
        let all = Schema::assertions("#/prefix", SchemaAssertions {
            types: Some(vec![SchemaType::Object,SchemaType::Array]),
            const_value: Some(json!({"first":-0.0,"second":"한글"})),
            enum_values: Some(vec![json!(1),json!("a")]),
            object: Some(ObjectSchema { properties: vec![PropertySchema {name:"a".into(),schema:schemas[4].clone()}],
                pattern_properties: vec![PatternPropertySchema { pattern:"^x".into(),schema:schemas[5].clone()}],
                property_names: Some(schemas[4].clone()),additional_properties:AdditionalProperties::Schema(Box::new(schemas[5].clone())),
                ..Default::default() }),
            array: Some(ArraySchema {items:Box::new(schemas[4].clone()),prefix_items:schemas[..3].to_vec(),min_items:2,max_items:Some(8)}),
            string:Some(StringSchema::default()),number:Some(NumberSchema::default()),
            any_of:schemas[..3].to_vec(),one_of:schemas[3..6].to_vec(),all_of:schemas[6..9].to_vec(),not:Some(schemas[4].clone()),
        });
        schemas.push(all);
        let stack = BTreeSet::from(["#".to_owned(),"#/한글".to_owned()]);
        for schema in &schemas {
            for class in [JsonTerminalPartitionClass::Other,JsonTerminalPartitionClass::Literal,JsonTerminalPartitionClass::Pattern] {
                for site in [StructuralSchemaSite::Ordinary,StructuralSchemaSite::AdditionalProperties] {
                    let query = BorrowedKey {schema,terminal_partition_class:class,site,object_variant_ref_stack:&stack};
                    let reference=owned(&query);
                    assert_eq!(bincode::serialize(&query).unwrap(),bincode::serialize(&reference).unwrap());
                    assert_eq!(query.fingerprint(),reference.fingerprint());
                    assert!(query.matches(&reference));
                    for other in &schemas {
                        let other=owned(&BorrowedKey {schema:other,..query});
                        assert_eq!(query.matches(&other),reference==other);
                    }
                    let mut altered=reference.clone();altered.object_variant_ref_stack.push("extra".into());
                    assert!(!query.matches(&altered));
                    altered=reference.clone();altered.site=match site {StructuralSchemaSite::Ordinary=>StructuralSchemaSite::AdditionalProperties,_=>StructuralSchemaSite::Ordinary};
                    assert!(!query.matches(&altered));
                }
            }
        }
    }
}
